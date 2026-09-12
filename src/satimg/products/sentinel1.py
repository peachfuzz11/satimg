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

from satimg import doppler
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
    effect for an object detected in the image.
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

    @cached_property
    def raw(self) -> xarray.DataArray:
        measurement = os.path.join(self._path, "measurement")
        pols = {}
        for f in os.listdir(measurement):
            m = re.search(r"-(vv|vh|hh|hv)-", f)
            if m:
                pols[os.path.join(measurement, f)] = m.group(1)
        files = sorted(pols, key=lambda f: _POL_ORDER[pols[f]])
        merged = merge_bands(files)
        da = label_bands(as_band_yx(merged), [pols[f].upper() for f in files])
        return keep_open(da, merged)

    def _render_visual(self) -> xarray.DataArray:
        a = self.raw
        db = (10 * numpy.log10(a.where(a > 0))).fillna(0).mean("band")
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

    def heading_to_los(self, heading_deg):
        """Convert a compass heading (degrees clockwise from true north) into the
        object's bearing relative to this scene's radar line of sight. See
        :func:`satimg.doppler.heading_to_los` for the convention and equations."""
        rel = doppler.heading_to_los(heading_deg, self.metadata.attrs["platform_heading"])
        return float(rel) if numpy.ndim(heading_deg) == 0 else numpy.asarray(rel)

    def doppler_azimuth_shift(self, rowcol, speed: float, heading_deg: float):
        """Azimuth-direction pixel displacement of a moving object at ``rowcol``.

        ``rowcol`` is a ``(row, col)`` pixel pair (or a list of pairs / an ``(N, 2)``
        array, matching :meth:`~satimg.metadata.Field.at`), ``speed`` is the
        object's ground speed in m/s, and ``heading_deg`` its compass heading in
        degrees clockwise from true north. Returns the estimated shift in pixels
        along the azimuth (row) axis -- positive towards higher row indices. See
        :func:`satimg.doppler.azimuth_shift_m` for the underlying physics and sign
        convention.
        """
        m = self.metadata
        attrs = m.attrs
        incidence = m.incidence_angle.at(rowcol)
        slant_range_m = m.slant_range_time.at(rowcol) * doppler.SPEED_OF_LIGHT / 2
        shift_m = doppler.azimuth_shift_m(
            speed, heading_deg, attrs["platform_heading"], incidence,
            slant_range_m, attrs["platform_velocity"],
        )
        shift_px = shift_m / attrs["azimuth_pixel_spacing"]
        return float(shift_px) if numpy.ndim(rowcol) == 1 else numpy.asarray(shift_px)
