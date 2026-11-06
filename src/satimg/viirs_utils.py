"""VIIRS Day/Night Band parsing, in both of the formats it is distributed in:

* NOAA's SDR (``SVDNB_*.h5``) + geolocation (``GDNBO_*.h5``), JPSS HDF5 --
  on NOAA's open AWS buckets;
* NASA's L1B (``VNP02DNB`` / ``VJ102DNB`` / ``VJ202DNB``, ``*.nc``) +
  geolocation (``V??03DNB``), netCDF4 -- from LAADS DAAC, Earthdata login.

Both give the same calibrated radiance (W / cm^2 sr) and per-pixel geolocation
and angles; the readers here bring them to one shape: ``read_*_granule`` ->
timestamp / footprint / attrs, ``read_*_geolocation`` -> ``lat`` / ``lon`` /
:data:`FIELDS` / moon. Pure functions over ``h5py`` files; see
:mod:`satimg.products.viirs` for the products.
"""

from __future__ import annotations

import datetime
import re

import numpy

from satimg import swath
from satimg.source import Source

#: detector rows per VIIRS scan: geolocation is linear in row within one,
#: but jumps between scans (the "bow-tie" overlap at the swath edges)
SCAN_ROWS = 16

#: anything at or below this is one of the float fill values (-999.3, -999.9, ...)
FILL_BELOW = -999.0

#: the per-pixel metadata fields, all in degrees
FIELDS = ("satellite_zenith", "satellite_azimuth", "solar_zenith", "solar_azimuth", "lunar_zenith", "lunar_azimuth")

#: JPSS platform code -> satellite
PLATFORMS = {"npp": "snpp", "j01": "noaa-20", "j02": "noaa-21", "j03": "noaa-22", "j04": "noaa-23"}

# -- NOAA SDR ----------------------------------------------------------------------
SDR = "All_Data/VIIRS-DNB-SDR_All"
GEO = "All_Data/VIIRS-DNB-GEO_All"
GRANULE = "Data_Products/VIIRS-DNB-SDR/VIIRS-DNB-SDR_Gran_0"
SDR_RADIANCE = f"{SDR}/Radiance"

#: GDNBO dataset -> field
_SDR_ANGLES = {
    "SatelliteZenithAngle": "satellite_zenith",
    "SatelliteAzimuthAngle": "satellite_azimuth",
    "SolarZenithAngle": "solar_zenith",
    "SolarAzimuthAngle": "solar_azimuth",
    "LunarZenithAngle": "lunar_zenith",
    "LunarAzimuthAngle": "lunar_azimuth",
}

_NAME = re.compile(
    r"^(?P<kind>SVDNB|GDNBO)_(?P<sat>\w+?)_d(?P<date>\d{8})_t(?P<start>\d{7})_e(?P<end>\d{7})_b(?P<orbit>\d+)_"
)


def granule_times(name: str) -> tuple[datetime.datetime, datetime.datetime]:
    """A JPSS file name's start / end time (``_dYYYYMMDD_tHHMMSSs_eHHMMSSs``,
    tenths of a second; an end before the start rolls over midnight)."""
    m = _NAME.match(name)
    if not m:
        raise ValueError(f"not a VIIRS DNB file name: {name!r}")
    day = datetime.datetime.strptime(m["date"], "%Y%m%d")

    def at(hhmmsss: str) -> datetime.datetime:
        return day + datetime.timedelta(
            hours=int(hhmmsss[:2]), minutes=int(hhmmsss[2:4]), seconds=int(hhmmsss[4:]) / 10
        )

    start, end = at(m["start"]), at(m["end"])
    return start, end if end >= start else end + datetime.timedelta(days=1)


def granule_key(name: str) -> str:
    """What pairs an SDR with its GEO: platform, date, start, end, orbit."""
    m = _NAME.match(name)
    return f"{m['sat']}_d{m['date']}_t{m['start']}_e{m['end']}_b{m['orbit']}"


def _pick(source: Source, prefix: str, suffix: str) -> str:
    """The source itself if it is one ``prefix*suffix`` file, else the first
    such file at its root."""
    if source.name.startswith(prefix) and source.name.endswith(suffix):
        return source.name
    names = sorted(n for n in source.listdir() if n.startswith(prefix) and n.endswith(suffix))
    if not names:
        raise FileNotFoundError(f"no {prefix}*{suffix} under {source.name}")
    return names[0]


def find_sdr_pair(source: Source) -> tuple[str, str]:
    """Root-relative ``(sdr, geo)``: the ``SVDNB`` (the source itself, or at
    its root) and the ``GDNBO`` of the same granule beside it."""
    sdr = _pick(source, "SVDNB_", ".h5")
    key = granule_key(sdr)
    geos = sorted(n for n in source.listdir() if n.startswith("GDNBO_") and n.endswith(".h5") and granule_key(n) == key)
    if not geos:
        raise FileNotFoundError(f"no GDNBO geolocation file next to {sdr}")
    return sdr, geos[0]


def _attr(attrs, name):
    value = numpy.asarray(attrs[name]).ravel()[0]
    return value.decode() if isinstance(value, bytes) else value


def _when(date: str, time: str) -> datetime.datetime:
    # e.g. "20261101", "163006.418783Z"
    return datetime.datetime.strptime(f"{date}{time.rstrip('Z')}", "%Y%m%d%H%M%S.%f")


def read_sdr_granule(sdr_h5) -> dict:
    """Timestamp (granule centre), GeoJSON footprint (the G-ring) and
    scene-wide attrs from the SDR granule's own attributes."""
    attrs = sdr_h5[GRANULE].attrs
    start = _when(_attr(attrs, "Beginning_Date"), _attr(attrs, "Beginning_Time"))
    stop = _when(_attr(attrs, "Ending_Date"), _attr(attrs, "Ending_Time"))
    lats, lons = attrs["G-Ring_Latitude"].ravel(), attrs["G-Ring_Longitude"].ravel()
    ring = [(float(lon), float(lat)) for lat, lon in zip(lats, lons) if lat > FILL_BELOW]
    platform = _attr(sdr_h5.attrs, "Platform_Short_Name").lower()
    return {
        "timestamp": start + (stop - start) / 2,
        "footprint": {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring + ring[:1]]}},
        "attrs": {
            "satellite": PLATFORMS.get(platform, platform),
            "orbit": int(_attr(attrs, "N_Beginning_Orbit_Number")),
            "day_night": _attr(attrs, "N_Day_Night_Flag").lower(),
            "pass": "descending" if int(_attr(attrs, "Ascending/Descending_Indicator")) else "ascending",
        },
    }


def read_sdr_geolocation(geo_h5) -> dict:
    """The GDNBO file's full-resolution latitude / longitude (ellipsoid: the
    sea surface, where boats are) and :data:`FIELDS`, fills as ``NaN``."""
    group = geo_h5[GEO]
    out = {
        "lat": swath.read_scaled(group["Latitude"], FILL_BELOW),
        "lon": swath.read_scaled(group["Longitude"], FILL_BELOW),
    }
    for dataset, field in _SDR_ANGLES.items():
        out[field] = swath.read_scaled(group[dataset], FILL_BELOW)
    out["moon_illumination"] = float(group["MoonIllumFraction"][()].ravel()[0]) / 100.0
    out["moon_phase_angle"] = float(group["MoonPhaseAngle"][()].ravel()[0])
    return out


# -- NASA L1B ------------------------------------------------------------------------
L1B_RADIANCE = "observation_data/DNB_observations"

#: VNP03DNB geolocation_data variable -> field
_L1B_ANGLES = {
    "sensor_zenith": "satellite_zenith",
    "sensor_azimuth": "satellite_azimuth",
    "solar_zenith": "solar_zenith",
    "solar_azimuth": "solar_azimuth",
    "lunar_zenith": "lunar_zenith",
    "lunar_azimuth": "lunar_azimuth",
}

#: L1B short-name prefix -> satellite
L1B_PLATFORMS = {"VNP": "snpp", "VJ1": "noaa-20", "VJ2": "noaa-21", "VJ3": "noaa-22"}

_L1B_NAME = re.compile(
    r"^(?P<platform>VNP|VJ\d)0(?P<kind>[23])DNB(?P<nrt>_NRT)?\.A(?P<date>\d{7})\.(?P<time>\d{4})\.(?P<collection>\d{3})"
)


def l1b_key(name: str) -> str:
    """What pairs a V??02DNB with its V??03DNB: platform, date, time,
    collection, NRT or not."""
    m = _L1B_NAME.match(name)
    if not m:
        raise ValueError(f"not a VIIRS L1B DNB file name: {name!r}")
    return f"{m['platform']}{m['nrt'] or ''}.A{m['date']}.{m['time']}.{m['collection']}"


def find_l1b_pair(source: Source) -> tuple[str, str]:
    """Root-relative ``(l1b, geo)``: the ``V??02DNB`` (the source itself, or
    at its root) and the ``V??03DNB`` of the same granule beside it."""
    names = source.listdir()
    if _L1B_NAME.match(source.name) and source.name.endswith(".nc"):
        l1b = source.name
    else:
        l1bs = sorted(n for n in names if n.endswith(".nc") and (m := _L1B_NAME.match(n)) and m["kind"] == "2")
        if not l1bs:
            raise FileNotFoundError(f"no V??02DNB*.nc under {source.name}")
        l1b = l1bs[0]
    key = l1b_key(l1b)
    geos = sorted(
        n for n in names
        if n.endswith(".nc") and (m := _L1B_NAME.match(n)) and m["kind"] == "3" and l1b_key(n) == key
    )
    if not geos:
        raise FileNotFoundError(f"no V??03DNB geolocation file next to {l1b}")
    return l1b, geos[0]


def _iso(text: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(text.rstrip("Z"))


def read_l1b_granule(l1b_h5, name: str) -> dict:
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
            "satellite": L1B_PLATFORMS.get(_L1B_NAME.match(name)["platform"], "unknown"),
            "orbit": int(orbit),
            "day_night": _attr(attrs, "DayNightFlag").lower(),
            "pass": _attr(attrs, "startDirection").lower(),
        },
    }


def read_l1b_geolocation(geo_h5) -> dict:
    """The V??03DNB file's latitude / longitude (terrain corrected -- the
    ellipsoid at sea) and :data:`FIELDS`, scaled, fills as ``NaN``. The moon
    illumination is per pixel there; its median stands for the granule."""
    group = geo_h5["geolocation_data"]
    out = {
        "lat": swath.read_scaled(group["latitude"], FILL_BELOW),
        "lon": swath.read_scaled(group["longitude"], FILL_BELOW),
    }
    for variable, field in _L1B_ANGLES.items():
        out[field] = swath.read_scaled(group[variable])
    # percent, per pixel
    out["moon_illumination"] = float(numpy.nanmedian(swath.read_scaled(group["moon_illumination_fraction"]))) / 100.0
    out["moon_phase_angle"] = (
        float(numpy.nanmedian(swath.read_scaled(group["moon_phase_angle"]))) if "moon_phase_angle" in group else float("nan")
    )
    return out


# -- both ---------------------------------------------------------------------------
def nodes(geolocation: dict, tolerance_m: float) -> dict:
    """A node grid for :class:`~satimg.transform.SwathTransformer` and the
    angle fields: scan-edge rows, :func:`~satimg.swath.node_columns` at
    ``tolerance_m``, as pixel-centre coordinates (``i + 0.5``), with each
    array sampled there."""
    lat, lon = geolocation["lat"], geolocation["lon"]
    rows = swath.scan_edge_rows(lat.shape[0], SCAN_ROWS)
    cols = swath.node_columns(lat[rows], lon[rows], tolerance_m)
    sample = numpy.ix_(rows, cols)
    out = {"rows": rows + 0.5, "cols": cols + 0.5, "lat": lat[sample], "lon": lon[sample]}
    for field in FIELDS:
        out[field] = geolocation[field][sample]
    return out


def gdal_path(raster_path: str, dataset: str) -> str:
    return f'HDF5:"{raster_path}"://{dataset}'
