"""VIIRS Day/Night Band SDR + GEO (JPSS ``SVDNB`` / ``GDNBO`` HDF5) parsing.

A granule is two files: the SDR (``SVDNB_*.h5``, calibrated radiance) and its
geolocation (``GDNBO_*.h5``, a latitude / longitude / angle for every pixel).
Pure functions over ``h5py`` files and plain arrays; see
:class:`~satimg.products.viirs.ViirsDNBProduct` for the product.
"""

from __future__ import annotations

import datetime
import re

import numpy

from satimg.source import Source

SDR = "All_Data/VIIRS-DNB-SDR_All"
GEO = "All_Data/VIIRS-DNB-GEO_All"
GRANULE = "Data_Products/VIIRS-DNB-SDR/VIIRS-DNB-SDR_Gran_0"

#: detector rows per VIIRS scan: geolocation is linear in row within one,
#: but jumps between scans (the "bow-tie" overlap at the swath edges)
SCAN_ROWS = 16

#: anything at or below this is one of the JPSS float fill values (-999.3, ...)
FILL_BELOW = -999.0

#: GEO dataset -> (metadata field name, units)
ANGLE_FIELDS = {
    "SatelliteZenithAngle": ("satellite_zenith", "degrees"),
    "SatelliteAzimuthAngle": ("satellite_azimuth", "degrees"),
    "SolarZenithAngle": ("solar_zenith", "degrees"),
    "SolarAzimuthAngle": ("solar_azimuth", "degrees"),
    "LunarZenithAngle": ("lunar_zenith", "degrees"),
    "LunarAzimuthAngle": ("lunar_azimuth", "degrees"),
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


#: JPSS platform code -> satellite
PLATFORMS = {"npp": "snpp", "j01": "noaa-20", "j02": "noaa-21", "j03": "noaa-22", "j04": "noaa-23"}


def find_pair(source: Source) -> tuple[str, str]:
    """Root-relative ``(sdr, geo)`` paths: the source itself (one ``SVDNB``
    file) or the ``SVDNB`` at its root (a folder / zip holding both); the
    GEO by the name the SDR's ``N_GEO_Ref`` gives, falling back to any
    ``GDNBO`` of the same granule (platform, date, start, end, orbit)."""
    names = source.listdir()
    if source.name.startswith("SVDNB_") and source.name.endswith(".h5"):
        sdr = source.name
    else:
        sdrs = sorted(n for n in names if n.startswith("SVDNB_") and n.endswith(".h5"))
        if not sdrs:
            raise FileNotFoundError(f"no SVDNB_*.h5 under {source.name}")
        sdr = sdrs[0]
    key = granule_key(sdr)
    geos = sorted(n for n in names if n.startswith("GDNBO_") and n.endswith(".h5") and granule_key(n) == key)
    if not geos:
        raise FileNotFoundError(f"no GDNBO geolocation file next to {sdr}")
    return sdr, geos[0]


def _attr(attrs, name):
    value = attrs[name].ravel()[0]
    return value.decode() if isinstance(value, bytes) else value


def _when(date: str, time: str) -> datetime.datetime:
    # e.g. "20261101", "163006.418783Z"
    return datetime.datetime.strptime(f"{date}{time.rstrip('Z')}", "%Y%m%d%H%M%S.%f")


def read_granule(sdr_h5) -> dict:
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


def _unit(lat, lon) -> numpy.ndarray:
    lat, lon = numpy.deg2rad(lat), numpy.deg2rad(lon)
    return numpy.stack([numpy.cos(lat) * numpy.cos(lon), numpy.cos(lat) * numpy.sin(lon), numpy.sin(lat)])


def scan_edge_rows(height: int) -> numpy.ndarray:
    """The first and last row of every scan: within a scan, geolocation is
    linear in row, so these are the only rows a node grid needs."""
    starts = numpy.arange(0, height, SCAN_ROWS)
    return numpy.unique(numpy.r_[starts, numpy.minimum(starts + SCAN_ROWS - 1, height - 1)])


def node_columns(lat: numpy.ndarray, lon: numpy.ndarray, tolerance_m: float) -> numpy.ndarray:
    """The fewest columns (greedily) between which linear interpolation of
    ``lat`` / ``lon`` (rows, all columns) stays within ``tolerance_m`` metres
    of the truth on every row given. The DNB's ground sampling changes in
    steps across the swath (its aggregation zones), so evenly spaced columns
    need several times more nodes for the same accuracy."""
    width = lat.shape[1]
    truth = _unit(lat, lon)  # (3, rows, width)
    nodes = numpy.array([0, width - 1])
    every = numpy.arange(width)
    while True:
        k = numpy.clip(numpy.searchsorted(nodes, every, side="right") - 1, 0, len(nodes) - 2)
        t = (every - nodes[k]) / (nodes[k + 1] - nodes[k])
        guess = truth[:, :, nodes[k]] * (1 - t) + truth[:, :, nodes[k + 1]] * t
        guess /= numpy.linalg.norm(guess, axis=0)
        error = numpy.nan_to_num(numpy.linalg.norm(guess - truth, axis=0) * 6_371_000.0).max(axis=0)
        bad = numpy.flatnonzero(error > tolerance_m)
        if not len(bad):
            return nodes
        # the worst column in every interval still out of tolerance
        order = bad[numpy.lexsort((-error[bad], k[bad]))]
        _, first = numpy.unique(k[order], return_index=True)
        nodes = numpy.union1d(nodes, order[first])


def read_geolocation(geo_h5) -> dict:
    """The GEO file's full-resolution latitude / longitude (ellipsoid: the
    sea surface, where boats are) and per-pixel angles, fills as ``NaN``."""
    group = geo_h5[GEO]

    def read(name):
        data = group[name][()].astype(float)
        data[data <= FILL_BELOW] = numpy.nan
        return data

    out = {"lat": read("Latitude"), "lon": read("Longitude")}
    for name in ANGLE_FIELDS:
        out[name] = read(name)
    out["moon_illumination"] = float(group["MoonIllumFraction"][()].ravel()[0]) / 100.0
    out["moon_phase_angle"] = float(group["MoonPhaseAngle"][()].ravel()[0])
    return out


def nodes(geolocation: dict, tolerance_m: float) -> dict:
    """A node grid for :class:`~satimg.transform.SwathTransformer` and the
    angle fields: scan-edge rows, :func:`node_columns` at ``tolerance_m``,
    as pixel-centre coordinates (``i + 0.5``), with each array sampled there."""
    lat, lon = geolocation["lat"], geolocation["lon"]
    rows = scan_edge_rows(lat.shape[0])
    cols = node_columns(lat[rows], lon[rows], tolerance_m)
    sample = numpy.ix_(rows, cols)
    out = {"rows": rows + 0.5, "cols": cols + 0.5, "lat": lat[sample], "lon": lon[sample]}
    for name in ANGLE_FIELDS:
        out[name] = geolocation[name][sample]
    return out


def radiance_path(raster_path: str) -> str:
    return f'HDF5:"{raster_path}"://{SDR}/Radiance'

