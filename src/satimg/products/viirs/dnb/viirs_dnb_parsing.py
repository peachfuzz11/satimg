"""VIIRS Day/Night Band L1B parsing: NASA's calibrated radiance plus its
per-pixel geolocation, netCDF4 -- ``VJ102DNB_NRT`` + ``VJ103DNB_NRT``
(NOAA-20) and ``VJ202DNB_NRT`` + ``VJ203DNB_NRT`` (NOAA-21) from LANCE, or
the standard-archive pairs of the same layout from LAADS.

Paths, types, scaling and fills follow the LAADS VNP02DNB / VNP03DNB file
specifications. Pure functions over ``h5py`` files; see
:class:`~satimg.products.viirs.dnb.viirs_dnb.ViirsDNB` for the product.
"""

from __future__ import annotations

import datetime
import re

import numpy

from satimg.products import swath_nodes
from satimg.readers.hdf5 import read_scaled
from satimg.source.source import Source

#: detector rows per VIIRS scan: geolocation is linear in row within one,
#: but jumps between scans (the "bow-tie" overlap at the swath edges)
SCAN_ROWS = 16

#: anything at or below this is one of the float fill values (-999.3, -999.9, ...)
FILL_BELOW = -999.0

#: the per-pixel metadata fields, all in degrees
FIELDS = ("satellite_zenith", "satellite_azimuth", "solar_zenith", "solar_azimuth", "lunar_zenith", "lunar_azimuth")

RADIANCE = "observation_data/DNB_observations"

#: VNP03DNB geolocation_data variable -> field
_ANGLES = {
    "sensor_zenith": "satellite_zenith",
    "sensor_azimuth": "satellite_azimuth",
    "solar_zenith": "solar_zenith",
    "solar_azimuth": "solar_azimuth",
    "lunar_zenith": "lunar_zenith",
    "lunar_azimuth": "lunar_azimuth",
}

#: L1B short-name prefix -> satellite
PLATFORMS = {"VNP": "snpp", "VJ1": "noaa-20", "VJ2": "noaa-21", "VJ3": "noaa-22"}

_NAME = re.compile(
    r"^(?P<platform>VNP|VJ\d)0(?P<kind>[23])DNB(?P<nrt>_NRT)?\.A(?P<date>\d{7})\.(?P<time>\d{4})\.(?P<collection>\d{3})"
)


def granule_key(name: str) -> str:
    """What pairs a V??02DNB with its V??03DNB: platform, date, time,
    collection, NRT or not."""
    m = _NAME.match(name)
    if not m:
        raise ValueError(f"not a VIIRS L1B DNB file name: {name!r}")
    return f"{m['platform']}{m['nrt'] or ''}.A{m['date']}.{m['time']}.{m['collection']}"


def find_pair(source: Source) -> tuple[str, str]:
    """Root-relative ``(l1b, geo)``: the ``V??02DNB`` (the source itself, or
    at its root) and the ``V??03DNB`` of the same granule beside it."""
    names = source.listdir()
    if _NAME.match(source.name) and source.name.endswith(".nc"):
        l1b = source.name
    else:
        l1bs = sorted(n for n in names if n.endswith(".nc") and (m := _NAME.match(n)) and m["kind"] == "2")
        if not l1bs:
            raise FileNotFoundError(f"no V??02DNB*.nc under {source.name}")
        l1b = l1bs[0]
    key = granule_key(l1b)
    geos = sorted(
        n for n in names
        if n.endswith(".nc") and (m := _NAME.match(n)) and m["kind"] == "3" and granule_key(n) == key
    )
    if not geos:
        raise FileNotFoundError(f"no V??03DNB geolocation file next to {l1b}")
    return l1b, geos[0]


def _attr(attrs, name):
    value = numpy.asarray(attrs[name]).ravel()[0]
    return value.decode() if isinstance(value, bytes) else value


def _iso(text: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(text.rstrip("Z"))


def read_granule(l1b_h5, name: str) -> dict:
    """Timestamp (coverage centre), footprint (the G-ring) and scene-wide
    attrs from the L1B file's global attributes."""
    attrs = l1b_h5.attrs
    start, stop = _iso(_attr(attrs, "time_coverage_start")), _iso(_attr(attrs, "time_coverage_end"))
    lats, lons = numpy.asarray(attrs["GRingPointLatitude"]).ravel(), numpy.asarray(attrs["GRingPointLongitude"]).ravel()
    ring = [(float(lon), float(lat)) for lat, lon in zip(lats, lons)]
    orbit = _attr(attrs, "orbit_number") if "orbit_number" in attrs else _attr(attrs, "OrbitNumber")
    return {
        "timestamp": start + (stop - start) / 2,
        "footprint": {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring + ring[:1]]}},
        "attrs": {
            "satellite": PLATFORMS.get(_NAME.match(name)["platform"], "unknown"),
            "orbit": int(orbit),
            "day_night": _attr(attrs, "DayNightFlag").lower(),
            "pass": _attr(attrs, "startDirection").lower(),
        },
    }


def read_geolocation(geo_h5) -> dict:
    """The V??03DNB file's latitude / longitude (terrain corrected -- the
    ellipsoid at sea), scaled, fills as ``NaN`` -- on its scan-edge rows
    only (``"rows"``), all a node grid needs (see :func:`nodes`): a quarter
    of the file's positions are never read whole, an eighth is. The moon
    illumination is per pixel there; the median of every
    :data:`SCAN_ROWS`-th row and column stands for the granule.
    ``"shape"`` is the full granule's."""
    group = geo_h5["geolocation_data"]
    shape = group["latitude"].shape
    rows = swath_nodes.scan_edge_rows(shape[0], SCAN_ROWS)
    sparse = (slice(None, None, SCAN_ROWS), slice(None, None, SCAN_ROWS))

    def median(variable):
        return float(numpy.nanmedian(read_scaled(group[variable], key=sparse)))

    return {
        "shape": shape,
        "rows": rows,
        "lat": read_scaled(group["latitude"], FILL_BELOW, rows),
        "lon": read_scaled(group["longitude"], FILL_BELOW, rows),
        # percent, per pixel
        "moon_illumination": median("moon_illumination_fraction") / 100.0,
        "moon_phase_angle": median("moon_phase_angle") if "moon_phase_angle" in group else None,
    }


def nodes(geolocation: dict, tolerance_m: float) -> dict:
    """A node grid for :class:`~satimg.transform.swath_transformer.SwathTransformer`:
    :func:`read_geolocation`'s scan-edge rows,
    :func:`~satimg.products.swath_nodes.node_columns` at ``tolerance_m``, as
    pixel-centre coordinates (``i + 0.5``), latitude / longitude sampled
    there; ``"row_index"`` / ``"col_index"`` are the integer pixels, for
    :func:`read_angles`."""
    lat, lon, rows = geolocation["lat"], geolocation["lon"], geolocation["rows"]
    cols = swath_nodes.node_columns(lat, lon, tolerance_m)
    return {
        "rows": rows + 0.5, "cols": cols + 0.5, "lat": lat[:, cols], "lon": lon[:, cols],
        "row_index": rows, "col_index": cols,
    }


def read_angles(geo_h5, node_grid: dict) -> dict:
    """:data:`FIELDS` at ``node_grid``'s (:func:`nodes`) pixels, scaled,
    fills as ``NaN`` -- read a variable's node rows at a time, never whole."""
    group = geo_h5["geolocation_data"]
    rows, cols = node_grid["row_index"], node_grid["col_index"]
    return {field: read_scaled(group[variable], key=rows)[:, cols] for variable, field in _ANGLES.items()}


def gdal_path(raster_path: str, dataset: str) -> str:
    return f'HDF5:"{raster_path}"://{dataset}'
