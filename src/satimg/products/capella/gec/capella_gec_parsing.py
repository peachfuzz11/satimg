"""Parsing a Capella GEC product folder.

A product, as Capella's open-data bucket lays it out, is a folder named for
it -- ``CAPELLA_C13_SP_GEC_HH_20260909150138_20260909150146/`` (satellite,
mode: ``SP`` spotlight / ``SM`` stripmap / ``SS`` sliding spotlight,
product type, polarisation, start and stop time) -- holding ``<name>.tif``
(the geocoded, ellipsoid-corrected amplitude image) and
``<name>_extended.json`` (Capella's full collect metadata). The quick-look
PNG is published with the sibling GEO product, as
``<name with GEO for GEC>_thumb.png``.
"""

from __future__ import annotations

import datetime
import json
import re

from satimg.source.source import Source

#: the folder (or ``.tif``) name
NAME = re.compile(
    r"^CAPELLA_C\d{2}_(?P<mode>SP|SM|SS)_GEC_(?P<pol>[HV]{2})_(?P<start>\d{14})_(?P<stop>\d{14})"
)


def stem(source: Source) -> str:
    """The product's file-name stem (its folder name, no ``.tif``)."""
    m = NAME.match(source.name)
    if m is None:
        raise ValueError(f"not a Capella GEC product: {source.name}")
    return m.group(0)


def find_files(source: Source) -> tuple[str, str, str | None]:
    """``(GEC raster, extended metadata JSON, quick-look PNG or None)``."""
    s = stem(source)
    tif, meta = f"{s}.tif", f"{s}_extended.json"
    for name in (tif, meta):
        if not source.exists(name):
            raise FileNotFoundError(f"no {name} under {source.name}")
    thumb = s.replace("_GEC_", "_GEO_") + "_thumb.png"
    return tif, meta, thumb if source.exists(thumb) else None


def _time(text: str) -> datetime.datetime:
    """A Capella UTC timestamp (nanoseconds, ``Z``) as a naive datetime."""
    text = text.rstrip("Z")
    head, _, frac = text.partition(".")
    t = datetime.datetime.fromisoformat(head)
    return t.replace(microsecond=int(frac[:6].ljust(6, "0"))) if frac else t


def read_metadata(source: Source, relpath: str) -> dict:
    """``{"timestamp", "scale_factor", "attrs"}`` from the extended JSON --
    ``timestamp`` the scene centre's time (naive UTC), ``scale_factor`` what
    turns a pixel value into amplitude (``sigma0 = (DN * scale_factor)^2``),
    ``attrs`` the scene-wide values below (angles in degrees, distances in
    metres)."""
    with source.open(relpath) as f:
        doc = json.load(f)
    collect = doc["collect"]
    image, radar = collect["image"], collect["radar"]
    centre = image["center_pixel"]
    scale = float(image["scale_factor"])
    return {
        "timestamp": _time(centre["center_time"]),
        "scale_factor": scale,
        "attrs": {
            "platform": collect.get("platform"),
            "mode": collect.get("mode"),
            "polarisations": [radar["transmit_polarization"] + radar["receive_polarization"]],
            "look_direction": radar.get("pointing"),
            "pass": collect.get("state", {}).get("direction"),
            "incidence_angle": centre.get("incidence_angle"),
            "look_angle": centre.get("look_angle"),
            "squint_angle": centre.get("squint_angle"),
            "radiometry": image.get("radiometry"),
            "scale_factor": scale,
            "resolution_range": image.get("ground_range_resolution"),
            "resolution_azimuth": image.get("ground_azimuth_resolution"),
            "looks": image.get("enl"),
            "pixel_spacing": [image.get("pixel_spacing_row"), image.get("pixel_spacing_column")],
            "collect_id": collect.get("collect_id"),
        },
    }
