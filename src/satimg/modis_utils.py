"""MODIS Collection 7 L1B 250 m (``MOD02QKM`` Terra / ``MYD02QKM`` Aqua)
parsing, with its ``MOD03`` / ``MYD03`` geolocation.

Collection 7 is netCDF4 / HDF-EOS5 (6.1 was HDF4, which this does not read).
Layout and semantics follow the MCST MODIS Level 1B Product User's Guide
(M1054 Rev E, C7): ``EV_250_RefSB`` -- uint16 scaled integers, bands 1 (red,
645 nm) and 2 (NIR, 858 nm), 40 rows per scan -- and ``Latitude`` /
``Longitude`` at 1 km (every 4th 250 m pixel, 10 rows per scan). Datasets are
found by name, not by group path, so the HDF-EOS5 group layout need not be
fixed here. The ``M?D03`` geolocation file of the same granule, when it is
alongside, gives the same 1 km latitude / longitude plus the sun and view
angles (:data:`ANGLES`).
"""

from __future__ import annotations

import datetime
import re

import numpy

from satimg import swath
from satimg.readers.hdf5 import find_dataset, read_scaled
from satimg.source.base import Source

DATA = "EV_250_RefSB"

#: 250 m rows per MODIS scan (10 detectors at 1 km, 40 at 250 m)
SCAN_ROWS = 40
#: 250 m pixels per 1 km geolocation pixel, each way
RATIO = 4

#: where the first 1 km geolocation pixel's centre sits from the first 250 m
#: pixel's centre, in 250 m pixels (track, scan): the MODIS "Fractional
#: Offset" the guide documents -- 1.5 down track -- used when the file does
#: not carry its own
DEFAULT_OFFSETS = (1.5, 0.0)

#: valid scaled integers; above this are fills / flags (65535 fill, ...)
VALID_MAX = 32767

BANDS = ("B01", "B02")
WAVELENGTHS_NM = (645.0, 858.5)

#: M?D03 dataset -> (metadata field, units); int16 x 0.01 in the file
ANGLES = {
    "SolarZenith": ("solar_zenith", "degrees"),
    "SolarAzimuth": ("solar_azimuth", "degrees"),
    "SensorZenith": ("satellite_zenith", "degrees"),
    "SensorAzimuth": ("satellite_azimuth", "degrees"),
}

#: short-name prefix -> satellite
PLATFORMS = {"MOD": "terra", "MYD": "aqua"}

_NAME = re.compile(
    r"^(?P<platform>MOD|MYD)02QKM(?P<nrt>_NRT)?\.A(?P<date>\d{7})\.(?P<time>\d{4})\.(?P<collection>\d{3})"
)
_GEO_NAME = re.compile(
    r"^(?P<platform>MOD|MYD)03(?P<nrt>_NRT)?\.A(?P<date>\d{7})\.(?P<time>\d{4})\.(?P<collection>\d{3})"
)


def _key(m) -> str:
    return f"{m['platform']}{m['nrt'] or ''}.A{m['date']}.{m['time']}.{m['collection']}"

#: a MODIS L1B granule is five minutes of orbit
GRANULE = datetime.timedelta(minutes=5)


def parse_name(name: str) -> dict:
    m = _NAME.match(name)
    if not m:
        raise ValueError(f"not a MODIS 250 m L1B file name: {name!r}")
    start = datetime.datetime.strptime(m["date"] + m["time"], "%Y%j%H%M")
    return {"satellite": PLATFORMS[m["platform"]], "start": start, "collection": m["collection"]}


def find_file(source: Source) -> str:
    """Root-relative path of the granule's ``.nc``: the source itself, or the
    one at its root (a folder / zip holding it)."""
    if _NAME.match(source.name) and source.name.endswith(".nc"):
        return source.name
    names = sorted(n for n in source.listdir() if _NAME.match(n) and n.endswith(".nc"))
    if not names:
        hdf = [n for n in source.listdir() if _NAME.match(n) and n.endswith(".hdf")]
        if hdf or source.name.endswith(".hdf"):
            raise ValueError(
                f"{source.name}: a Collection 6.1 HDF4 granule -- satimg reads Collection 7 (netCDF4) only"
            )
        raise FileNotFoundError(f"no M?D02QKM*.nc under {source.name}")
    return names[0]


def find_geolocation(source: Source, data_file: str) -> str | None:
    """Root-relative path of the ``M?D03`` of the same granule as
    ``data_file`` beside it, or ``None``."""
    key = _key(_NAME.match(data_file))
    for name in sorted(source.listdir()):
        m = _GEO_NAME.match(name)
        if m and name.endswith(".nc") and _key(m) == key:
            return name
    return None


def _attr(attrs, *names, default=None):
    for name in names:
        if name in attrs:
            value = numpy.asarray(attrs[name]).ravel()[0]
            return value.decode() if isinstance(value, bytes) else value
    return default


def fractional_offsets(h5) -> tuple[float, float]:
    """(track, scan) offsets of the 1 km geolocation from the 250 m data, in
    250 m pixels: the file's own ``...FractionalOffset...`` attributes where
    it has them (plus the integer HDF-EOS offset, 0 for MODIS), else
    :data:`DEFAULT_OFFSETS`."""
    track, scan = DEFAULT_OFFSETS
    for name, value in h5.attrs.items():
        if "FractionalOffset" not in name:
            continue
        value = float(numpy.asarray(value).ravel()[0])
        if "10*nscans" in name:
            track = value
        elif "Max_EV_frames" in name:
            scan = value
    return track, scan


def read_granule(h5, name: str) -> dict:
    """Timestamp, scene attrs and the radiometric scaling of each band."""
    info = parse_name(name)
    attrs = h5.attrs
    data = h5[find_dataset(h5, DATA)]
    day_night = _attr(attrs, "DAYNIGHTFLAG", "DayNightFlag", default="unknown")
    return {
        "timestamp": info["start"] + GRANULE / 2,
        "attrs": {
            "satellite": info["satellite"],
            "collection": info["collection"],
            "day_night": str(day_night).lower(),
            "reflectance_scales": [float(v) for v in numpy.ravel(data.attrs["reflectance_scales"])],
            "reflectance_offsets": [float(v) for v in numpy.ravel(data.attrs["reflectance_offsets"])],
            "radiance_scales": [float(v) for v in numpy.ravel(data.attrs["radiance_scales"])],
            "radiance_offsets": [float(v) for v in numpy.ravel(data.attrs["radiance_offsets"])],
        },
        "shape": tuple(data.shape[1:]),
    }


def read_geolocation(h5) -> dict:
    """The 1 km ``Latitude`` / ``Longitude``, fills as ``NaN`` -- on their
    scan-edge rows only (``"rows"``), all a node grid needs (see
    :func:`nodes`), so the rest is never read. ``"shape"`` is the 1 km
    grid's."""
    lat = h5[find_dataset(h5, "Latitude")]
    rows = swath.scan_edge_rows(lat.shape[0], SCAN_ROWS // RATIO)
    return {
        "shape": lat.shape,
        "rows": rows,
        "lat": read_scaled(lat, -999.0, rows),
        "lon": read_scaled(h5[find_dataset(h5, "Longitude")], -999.0, rows),
    }


def nodes(geolocation: dict, offsets: tuple[float, float], tolerance_m: float) -> dict:
    """A :class:`~satimg.transform.swath.SwathTransformer` node grid in 250 m pixel
    coordinates: :func:`read_geolocation`'s scan-edge rows and
    :func:`~satimg.swath.node_columns`, each 1 km pixel placed at
    ``4 i + offset`` (pixel centres at ``+ 0.5``); ``"row_index"`` /
    ``"col_index"`` are the 1 km pixels, for :func:`read_angles`."""
    lat, lon, rows = geolocation["lat"], geolocation["lon"], geolocation["rows"]
    cols = swath.node_columns(lat, lon, tolerance_m)
    track, scan = offsets
    return {
        "rows": RATIO * rows + track + 0.5,
        "cols": RATIO * cols + scan + 0.5,
        "lat": lat[:, cols],
        "lon": lon[:, cols],
        "row_index": rows,
        "col_index": cols,
    }


def read_angles(h5, node_grid: dict) -> dict:
    """An ``M?D03`` file's :data:`ANGLES` fields at ``node_grid``'s
    (:func:`nodes`) pixels, scaled -- read a dataset's node rows at a time,
    never whole."""
    rows, cols = node_grid["row_index"], node_grid["col_index"]
    return {
        field: read_scaled(h5[find_dataset(h5, dataset)], key=rows)[:, cols]
        for dataset, (field, _units) in ANGLES.items()
    }
