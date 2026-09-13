"""Sentinel-1 GRD (IW and EW) reader."""

from __future__ import annotations

import datetime
import os
import re
from functools import cached_property
from xml.etree import ElementTree

import numpy
import PIL.Image
import rasterio
import xarray

from satimg import sar_utils
from satimg.metadata import Metadata, grid_from_points
from satimg.product import Product
from satimg.readers import find_file, keep_open, merge_bands
from satimg.registry import register
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


def _read_manifest(path: str) -> dict:
    manifest = find_file(path, "manifest.safe")
    if manifest is None:
        raise FileNotFoundError(f"no manifest.safe under {path}")
    root = ElementTree.parse(manifest).getroot()
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


def _annotation_file(path: str) -> str:
    """First product-annotation XML (the ``calibration/`` ones are excluded)."""
    ann = os.path.join(path, "annotation")
    files = sorted(f for f in os.listdir(ann) if f.endswith(".xml"))
    if not files:
        raise FileNotFoundError(f"no annotation XML under {ann}")
    return os.path.join(ann, files[0])


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


def _read_geolocation(xml_path: str, at: datetime.datetime) -> tuple[list[dict], dict]:
    root = ElementTree.parse(xml_path).getroot()
    points = []
    for gp in root.findall(".//geolocationGridPoint"):
        pt = {"row": int(gp.findtext("line")), "col": int(gp.findtext("pixel"))}
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
    return points, attrs


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

    def __init__(self, path: str):
        super().__init__(path)
        meta = _read_manifest(path)
        self._timestamp = meta["timestamp"]
        self._footprint = meta["footprint"]
        with rasterio.open(path) as src:
            gcps, crs = src.gcps
        self._transformer = GCPTransformer(gcps, crs)

    @property
    def mode(self) -> str:
        m = re.search(r"_(IW|EW)_", os.path.basename(self._path))
        return m.group(1) if m else "IW"

    def _open_raw(self, tile: int | tuple[int, int]) -> xarray.DataArray:
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
        db = (10 * numpy.log10(raw.where(raw > 0))).fillna(0).mean("band")
        u8 = (255 / (1 + numpy.exp(-((db - 20) * 0.18)))).clip(0, 255).astype("uint8")
        return label_bands(as_band_yx(u8), ("amplitude",))

    def _read_metadata(self) -> Metadata:
        points, attrs = _read_geolocation(_annotation_file(self._path), self._timestamp)
        keys = tuple(_GEOLOC_FIELDS)
        grid = grid_from_points(points, keys)
        fields = {
            name: self._field(
                grid["rows"], grid["cols"], grid[name],
                name=name, units=_GEOLOC_FIELDS[name][1],
            )
            for name in keys
        }
        return Metadata(fields, attrs)

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
            p = os.path.join(self._path, "preview", name)
            if os.path.isfile(p):
                return PIL.Image.open(p)
        raise FileNotFoundError(f"no preview image under {self._path}")

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
