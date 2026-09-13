"""Sentinel-1 GRD (IW and EW) reader."""

from __future__ import annotations

import datetime
import os
import re
from functools import cached_property
from xml.etree import ElementTree

import numpy
import PIL.Image
import xarray
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS

from satimg import sar_utils
from satimg.metadata import Metadata, grid_from_points
from satimg.product import Product
from satimg.readers import keep_open, merge_bands
from satimg.registry import register
from satimg.source import Source
from satimg.tiling import as_band_yx, label_bands
from satimg.transform import GCPTransformer

#: measurement-file polarisation -> sort order (co-pol before cross-pol).
_POL_ORDER = {"vv": 0, "vh": 1, "hh": 2, "hv": 3}

_GEOLOC_FIELDS = {
    "incidence_angle": ("incidenceAngle", "degrees"),
    "elevation_angle": ("elevationAngle", "degrees"),
    "slant_range_time": ("slantRangeTime", "seconds"),
    "height": ("height", "metres"),
}

_NS = {
    "safe": "http://www.esa.int/safe/sentinel-1.0",
    "gml": "http://www.opengis.net/gml",
}

#: Sentinel-1's sun-synchronous orbit inclination, degrees -- actively maintained,
#: so effectively constant across the whole constellation and mission (ESA mission
#: documentation). Used to derive the local ground-track heading at a target's own
#: latitude; see :func:`satimg.sar_utils.ground_track_heading`.
_ORBIT_INCLINATION_DEG = 98.1813


def _read_manifest(source: Source) -> dict:
    manifest = source.find_file("manifest.safe")
    if manifest is None:
        raise FileNotFoundError(f"no manifest.safe under {source.name}")
    with source.open(manifest) as f:
        root = ElementTree.parse(f).getroot()
    start = datetime.datetime.strptime(
        root.find(".//safe:startTime", _NS).text, "%Y-%m-%dT%H:%M:%S.%f"
    )
    stop = datetime.datetime.strptime(
        root.find(".//safe:stopTime", _NS).text, "%Y-%m-%dT%H:%M:%S.%f"
    )
    coords = [
        (float(lon), float(lat))
        for pair in root.find(".//gml:coordinates", _NS).text.split()
        for lat, lon in [pair.split(",")]
    ]
    return {
        "timestamp": start + (stop - start) / 2,
        "footprint": {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        },
    }


def _annotation_file(source: Source) -> str:
    """Root-relative path of the first product-annotation XML (the
    ``calibration/`` ones are excluded)."""
    files = sorted(f for f in source.listdir("annotation") if f.endswith(".xml"))
    if not files:
        raise FileNotFoundError(f"no annotation XML under {source.name}/annotation")
    return f"annotation/{files[0]}"


def _nearest_orbit_velocity(root, at: datetime.datetime) -> float:
    """Platform speed (m/s) from the ``<orbit>`` state vector closest to ``at``.
    Sentinel-1's orbit is near-circular, so a single vector's speed is
    representative of the whole (~25s) scene -- no need to interpolate."""
    best_dt, best_velocity = None, None
    for orb in root.findall(".//orbit"):
        t = datetime.datetime.fromisoformat(orb.findtext("time"))
        dt = abs((t - at).total_seconds())
        if best_dt is None or dt < best_dt:
            velocity = numpy.array([float(orb.findtext(f"velocity/{c}")) for c in "xyz"])
            best_dt, best_velocity = dt, velocity
    return float(numpy.linalg.norm(best_velocity))


def _read_geolocation(
    source: Source, relpath: str, at: datetime.datetime
) -> tuple[list[dict], dict, tuple[int, int]]:
    """The annotation XML's geolocation grid points (row/col plus each
    :data:`_GEOLOC_FIELDS` value, and -- for building GCPs, not exposed in
    :class:`~satimg.metadata.Metadata` -- latitude/longitude/height), the
    scene-wide attrs, and the full measurement grid's ``(lines, samples)``
    shape (``imageAnnotation/imageInformation``)."""
    with source.open(relpath) as f:
        root = ElementTree.parse(f).getroot()
    points = []
    for gp in root.findall(".//geolocationGridPoint"):
        pt = {
            "row": int(gp.findtext("line")),
            "col": int(gp.findtext("pixel")),
            "latitude": float(gp.findtext("latitude")),
            "longitude": float(gp.findtext("longitude")),
            "height": float(gp.findtext("height")),
        }
        for name, (tag, _units) in _GEOLOC_FIELDS.items():
            pt[name] = float(gp.findtext(tag))
        points.append(pt)
    attrs = {
        "incidence_angle_mid_swath": float(root.findtext(".//incidenceAngleMidSwath")),
        "platform_heading": float(root.findtext(".//platformHeading")),
        "pass": root.findtext(".//pass"),
        "platform_velocity": _nearest_orbit_velocity(root, at),
        "orbit_inclination": _ORBIT_INCLINATION_DEG,
        "azimuth_pixel_spacing": float(root.findtext(".//azimuthPixelSpacing")),
    }
    shape = (
        int(root.findtext(".//numberOfLines")),
        int(root.findtext(".//numberOfSamples")),
    )
    return points, attrs, shape


@register(r"^S1[ABCD]_(IW_GRDH|EW_GRDM)_1SD[HV]_\d{8}T\d{6}_\d{8}T\d{6}.*\.SAFE$")
class Sentinel1Product(Product):
    """Ground-range-detected Sentinel-1, either acquisition mode.

    ``raw`` holds the calibrated backscatter (one band per polarisation);
    ``visual`` is the usual dB -> sigmoid stretch as a single greyscale band.
    ``metadata`` carries the ``geolocationGridPoint`` fields (``incidence_angle``,
    ``elevation_angle``, ``slant_range_time``, ``height``). :meth:`heading_to_los`
    and :meth:`doppler_azimuth_shift` estimate the SAR moving-target azimuth-shift
    effect for an object detected in the image, in pixels; :meth:`heading_in_image`
    re-expresses a compass heading in this GRD product's own (rotated) pixel frame;
    :meth:`correct_position` combines both to recover a moving target's true
    lat/lon from its as-detected position.
    """

    def __init__(self, source: Source):
        super().__init__(source)
        meta = _read_manifest(self._source)
        self._timestamp = meta["timestamp"]
        self._footprint = meta["footprint"]
        annotation = _annotation_file(self._source)
        points, attrs, shape = _read_geolocation(
            self._source, annotation, self._timestamp
        )
        self._geoloc_points = points
        self._geoloc_attrs = attrs
        self._geoloc_shape = shape
        gcps = [
            GroundControlPoint(
                row=p["row"], col=p["col"], x=p["longitude"], y=p["latitude"], z=p["height"],
            )
            for p in points
        ]
        self._transformer = GCPTransformer(gcps, CRS.from_epsg(4326))

    @property
    def mode(self) -> str:
        m = re.search(r"_(IW|EW)_", os.path.basename(self._path))
        return m.group(1) if m else "IW"

    def _open_raw(self, tile: int | tuple[int, int]) -> xarray.DataArray:
        self._require_extracted("raw")
        measurement = os.path.join(self._path, "measurement")
        pols = {}
        for f in os.listdir(measurement):
            m = re.search(r"-(vv|vh|hh|hv)-", f)
            if m:
                pols[os.path.join(measurement, f)] = m.group(1)
        files = sorted(pols, key=lambda f: _POL_ORDER[pols[f]])
        merged = merge_bands(files, tile=tile)
        da = label_bands(as_band_yx(merged), [pols[f].upper() for f in files])
        return keep_open(da, merged)

    def _render_visual(
        self, raw: xarray.DataArray, tile: int | tuple[int, int]
    ) -> xarray.DataArray:
        self._require_extracted("visual")
        db = (10 * numpy.log10(raw.where(raw > 0))).fillna(0).mean("band")
        u8 = (255 / (1 + numpy.exp(-((db - 20) * 0.18)))).clip(0, 255).astype("uint8")
        return label_bands(as_band_yx(u8), ("amplitude",))

    def _read_metadata(self) -> Metadata:
        keys = tuple(_GEOLOC_FIELDS)
        grid = grid_from_points(self._geoloc_points, keys)
        fields = {
            name: self._field(
                grid["rows"], grid["cols"], grid[name],
                name=name, units=_GEOLOC_FIELDS[name][1],
            )
            for name in keys
        }
        return Metadata(fields, self._geoloc_attrs)

    def _metadata_grid_shape(self) -> tuple[int, int]:
        return self._geoloc_shape

    @property
    def transformer(self) -> GCPTransformer:
        return self._transformer

    @property
    def timestamp(self) -> datetime.datetime:
        return self._timestamp

    @property
    def footprint(self) -> dict:
        return self._footprint

    def thumbnail(self) -> PIL.Image.Image:
        for name in ("thumbnail.png", "quick-look.png"):
            relpath = f"preview/{name}"
            if self._source.exists(relpath):
                with self._source.open(relpath) as f:
                    img = PIL.Image.open(f)
                    img.load()
                    return img
        raise FileNotFoundError(f"no preview image under {self._source.name}")

    def _local_platform_heading(self, rowcol):
        """Satellite ground-track heading at ``rowcol``'s own latitude (rather
        than the single scene-wide ``platform_heading`` attr), via
        :func:`satimg.sar_utils.ground_track_heading`."""
        lat = self.transformer.rowcol_to_latlon(rowcol)[:, 0]
        ascending = self.metadata.attrs["pass"].lower() == "ascending"
        heading = sar_utils.ground_track_heading(
            lat, self.metadata.attrs["orbit_inclination"], ascending
        )
        return float(heading[0]) if numpy.ndim(rowcol) == 1 else heading

    def heading_to_los(self, rowcol, heading_deg):
        """Convert a compass heading (degrees clockwise from true north) into the
        object's bearing relative to the radar line of sight at ``rowcol``.

        ``rowcol`` is a ``(row, col)`` pixel pair (or a list of pairs / an
        ``(N, 2)`` array, matching :meth:`~satimg.metadata.Field.at`) -- needed
        because the local satellite heading varies (slightly) with latitude across
        a scene. See :func:`satimg.sar_utils.heading_to_los` for the convention and
        equations.
        """
        rel = sar_utils.heading_to_los(heading_deg, self._local_platform_heading(rowcol))
        return float(rel) if numpy.ndim(rowcol) == 1 else numpy.asarray(rel)

    def heading_in_image(self, rowcol, heading_deg):
        """Convert a compass heading (degrees clockwise from true north) into
        this GRD product's own pixel frame, at pixel ``rowcol``.

        Unlike an orthorectified product, GRD imagery is still in native
        sensor geometry: row increases with azimuth time, i.e. towards the
        direction the platform is heading (see the sign convention in
        :func:`satimg.sar_utils.azimuth_shift_m`), so the image's "up"
        (decreasing row) points the *opposite* way -- the local platform
        heading plus 180 degrees. This is why an ascending-pass Sentinel-1
        GRD scene looks upside-down (south-up) relative to a map, and a
        descending-pass one looks right-side up.

        ``0``/``360`` means the object points towards the top of the image
        (decreasing row), ``90`` towards the right -- handy for e.g. drawing a
        detected ship's heading as an arrow directly on the raster. This is a
        SAR-only concern: a map-projected, north-up optical product needs no
        such conversion at all.
        """
        up_heading = (self._local_platform_heading(rowcol) + 180.0) % 360.0
        result = sar_utils.heading_in_image(heading_deg, up_heading)
        return float(result) if numpy.ndim(rowcol) == 1 else numpy.asarray(result)

    def doppler_azimuth_shift(self, rowcol, speed: float, heading_deg: float):
        """Azimuth-direction pixel displacement of a moving object at ``rowcol``.

        ``rowcol`` is a ``(row, col)`` pixel pair (or a list of pairs / an ``(N, 2)``
        array, matching :meth:`~satimg.metadata.Field.at`), ``speed`` is the
        object's ground speed in m/s, and ``heading_deg`` its compass heading in
        degrees clockwise from true north. Returns the estimated shift in pixels
        along the azimuth (row) axis -- positive towards higher row indices. See
        :func:`satimg.sar_utils.azimuth_shift_m` for the underlying physics and sign
        convention.
        """
        m = self.metadata
        attrs = m.attrs
        incidence = m.incidence_angle.at(rowcol)
        slant_range_m = m.slant_range_time.at(rowcol) * sar_utils.SPEED_OF_LIGHT / 2
        shift_m = sar_utils.azimuth_shift_m(
            speed, heading_deg, self._local_platform_heading(rowcol), incidence,
            slant_range_m, attrs["platform_velocity"],
        )
        shift_px = shift_m / attrs["azimuth_pixel_spacing"]
        return float(shift_px) if numpy.ndim(rowcol) == 1 else numpy.asarray(shift_px)

    def correct_position(self, lat: float, lon: float, speed: float, heading_deg: float):
        """Recover a moving target's true ``(lat, lon)`` from its as-detected
        position in this GRD product.

        ``lat``/``lon`` is where the target was read off the image, and ``speed``
        (m/s) / ``heading_deg`` (degrees clockwise from true north) are the same
        inputs as :meth:`doppler_azimuth_shift`. The target's own motion displaces
        it along the azimuth (row) axis of the focused image by
        :meth:`doppler_azimuth_shift`'s pixel shift; this undoes that displacement
        to recover the position the target actually occupied at acquisition time.
        """
        rowcol = self.transformer.latlon_to_rowcol((lat, lon))[0]
        shift_px = self.doppler_azimuth_shift(rowcol, speed, heading_deg)
        corrected_rowcol = (rowcol[0] - shift_px, rowcol[1])
        corrected_lat, corrected_lon = self.transformer.rowcol_to_latlon(corrected_rowcol)[0]
        return float(corrected_lat), float(corrected_lon)
