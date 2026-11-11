"""Parsing an Umbra GEC collect folder.

A collect, as Umbra's open-data bucket lays it out, is a folder named for
its start time and satellite, ``2025-01-31-14-10-48_UMBRA-08/``, holding
``<folder>_GEC.tif`` (the geocoded, ellipsoid-corrected amplitude image) and
the collect's metadata in one of two generations:

* ``<folder>.stac.v2.json`` -- a STAC item (collects from about 2024 on);
* ``<folder>_METADATA.json`` -- Umbra's own JSON (earlier collects).

Both carry the acquisition time, footprint and imaging geometry;
:func:`read_metadata` reads either into the same shape.
"""

from __future__ import annotations

import datetime
import json
import re

from satimg.source.source import Source

#: the folder (or ``_GEC.tif``) name: start time and satellite
NAME = re.compile(r"^(?P<stamp>\d{4}(?:-\d{2}){5})_(?P<platform>UMBRA-\d{2})")


def stem(source: Source) -> str:
    """The collect's file-name stem, ``2025-01-31-14-10-48_UMBRA-08``."""
    m = NAME.match(source.name)
    if m is None:
        raise ValueError(f"not an Umbra collect: {source.name}")
    return m.group(0)


def find_files(source: Source) -> tuple[str, str]:
    """``(GEC raster, metadata JSON)`` of the collect, by name."""
    s = stem(source)
    tif = f"{s}_GEC.tif"
    if not source.exists(tif):
        raise FileNotFoundError(f"no {tif} under {source.name}")
    for meta in (f"{s}.stac.v2.json", f"{s}_METADATA.json"):
        if source.exists(meta):
            return tif, meta
    raise FileNotFoundError(f"no {s}.stac.v2.json or {s}_METADATA.json under {source.name}")


def folder_time(folder: str) -> datetime.datetime:
    """The start time a collect folder's name gives
    (``2025-01-31-14-10-46_UMBRA-08``), timezone-aware UTC."""
    stamp = NAME.match(folder)["stamp"]
    return datetime.datetime.strptime(stamp, "%Y-%m-%d-%H-%M-%S").replace(tzinfo=datetime.timezone.utc)


def _time(text: str) -> datetime.datetime:
    """A UTC timestamp as a naive datetime (satimg's convention)."""
    t = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    return t.astimezone(datetime.timezone.utc).replace(tzinfo=None) if t.tzinfo else t


def _feature(geometry: dict) -> dict:
    """A GeoJSON Feature of ``geometry`` with any height dropped."""
    rings = [[tuple(c[:2]) for c in ring] for ring in geometry["coordinates"]]
    return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": rings}}


def _lower(value) -> str | None:
    return value.lower() if isinstance(value, str) else value


def read_metadata(source: Source, relpath: str) -> dict:
    """``{"timestamp", "footprint", "attrs"}`` from either metadata file --
    ``timestamp`` the middle of the collect (naive UTC), ``footprint`` a
    GeoJSON Feature, ``attrs`` the scene-wide values below (angles in
    degrees, distances in metres)."""
    with source.open(relpath) as f:
        doc = json.load(f)
    if relpath.endswith(".stac.v2.json"):
        return _from_stac(doc)
    return _from_v1(doc)


def _from_stac(item: dict) -> dict:
    p = item["properties"]
    return {
        "timestamp": _time(p["datetime"]),
        "footprint": _feature(item["geometry"]),
        "attrs": {
            "platform": _lower(p.get("platform")),
            "mode": _lower(p.get("sar:instrument_mode")),
            "polarisations": list(p.get("sar:polarizations", [])),
            "look_direction": _lower(p.get("sar:observation_direction")),
            "pass": _lower(p.get("sat:orbit_state")),
            "incidence_angle": p.get("view:incidence_angle"),
            "squint_angle": p.get("umbra:squint_angle_degrees_off_broadside"),
            "slant_range": p.get("umbra:slant_range_meters"),
            "resolution_range": p.get("sar:resolution_range"),
            "resolution_azimuth": p.get("sar:resolution_azimuth"),
            "collect_id": p.get("umbra:collect_id"),
        },
    }


def _from_v1(doc: dict) -> dict:
    collect = doc["collects"][0]
    start, end = _time(collect["startAtUTC"]), _time(collect["endAtUTC"])
    gec = (doc.get("derivedProducts", {}).get("GEC") or [{}])[0]
    resolution = gec.get("groundResolution", {})
    return {
        "timestamp": start + (end - start) / 2,
        "footprint": _feature(collect["footprintPolygonLla"]),
        "attrs": {
            "platform": _lower(doc.get("umbraSatelliteName", "")).replace("_", "-") or None,
            "mode": _lower(doc.get("imagingMode")),
            "polarisations": list(collect.get("polarizations", [])),
            "look_direction": _lower(collect.get("observationDirection")),
            "pass": _lower(collect.get("satelliteTrack")),
            "incidence_angle": collect.get("angleIncidenceDegrees"),
            "squint_angle": collect.get("angleSquintDegrees"),
            "slant_range": collect.get("slantRangeMeters"),
            "resolution_range": resolution.get("rangeMeters"),
            "resolution_azimuth": resolution.get("azimuthMeters"),
            "collect_id": collect.get("id"),
        },
    }
