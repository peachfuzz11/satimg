"""A small synthetic Umbra GEC collect folder, laid out as the open-data
bucket serves it -- real GECs are hundreds of MB.

``2025-01-31-14-10-48_UMBRA-08/``: a ``_GEC.tif`` (``uint8``, EPSG:4326 on
a rotated grid, as Umbra's are, with overviews) and the metadata JSON --
the STAC (``stac.v2``) generation by default, Umbra's older
``_METADATA.json`` with ``v1=True``. Water everywhere (DN 2), one bright
'ship' at :data:`SHIP` (DN 250), a zero (no-data) border.
"""

from __future__ import annotations

import json

import numpy

NAME = "2025-01-31-14-10-48_UMBRA-08"
HEIGHT, WIDTH = 256, 256
SHIP = (128, 100)
#: a rotated geotransform in degrees (~0.2 m pixels), like a real GEC's
TRANSFORM = (-58.42, 1.07e-06, 2.13e-06, -34.61, 1.76e-06, -8.85e-07)
FOOTPRINT = [[-58.42, -34.61, 30.0], [-58.419, -34.61, 30.0], [-58.419, -34.611, 30.0],
             [-58.42, -34.611, 30.0], [-58.42, -34.61, 30.0]]

STAC = {
    "type": "Feature",
    "id": "aa0e01ff-fc18-4ce8-bea0-da7438274a68",
    "geometry": {"type": "Polygon", "coordinates": [FOOTPRINT]},
    "properties": {
        "datetime": "2025-01-31T14:10:48.900000Z",
        "platform": "Umbra-08",
        "sar:instrument_mode": "SPOTLIGHT",
        "sar:polarizations": ["VV"],
        "sar:observation_direction": "left",
        "sat:orbit_state": "descending",
        "view:incidence_angle": 32.8,
        "umbra:squint_angle_degrees_off_broadside": 16.1,
        "umbra:slant_range_meters": 599816.8,
        "sar:resolution_range": 0.35,
        "sar:resolution_azimuth": 0.35,
        "umbra:collect_id": "180a083c-e47a-47f8-a756-303e6e5ed972",
        "umbra:task_id": "0049d051-c279-4b42-8cfe-58bf673683f0",
    },
    "assets": {},
}

V1 = {
    "version": "1.0.0",
    "imagingMode": "SPOTLIGHT",
    "umbraSatelliteName": "UMBRA_08",
    "collects": [{
        "id": "180a083c-e47a-47f8-a756-303e6e5ed972",
        "startAtUTC": "2025-01-31T14:10:47.000000+00:00",
        "endAtUTC": "2025-01-31T14:10:50.800000+00:00",
        "polarizations": ["VV"],
        "angleIncidenceDegrees": 32.8,
        "angleSquintDegrees": 16.1,
        "slantRangeMeters": 599816.8,
        "satelliteTrack": "DESCENDING",
        "observationDirection": "LEFT",
        "footprintPolygonLla": {"type": "Polygon", "coordinates": [FOOTPRINT]},
    }],
    "derivedProducts": {"GEC": [{"groundResolution": {"azimuthMeters": 0.36, "rangeMeters": 0.37}}]},
}


def image() -> numpy.ndarray:
    data = numpy.full((HEIGHT, WIDTH), 2, dtype="uint8")
    data[SHIP[0] - 3: SHIP[0] + 4, SHIP[1] - 1: SHIP[1] + 2] = 250
    data[:8], data[-8:], data[:, :8], data[:, -8:] = 0, 0, 0, 0
    return data


def build(parent, *, v1: bool = False):
    """Write the collect folder under ``parent``; returns its path."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine

    folder = parent / NAME
    folder.mkdir()
    a, b, c, d, e, f = TRANSFORM[1], TRANSFORM[2], TRANSFORM[0], TRANSFORM[4], TRANSFORM[5], TRANSFORM[3]
    with rasterio.open(
        folder / f"{NAME}_GEC.tif", "w", driver="GTiff", width=WIDTH, height=HEIGHT, count=1,
        dtype="uint8", crs="EPSG:4326", transform=Affine(a, b, c, d, e, f), nodata=0,
        tiled=True, blockxsize=128, blockysize=128,
    ) as dst:
        dst.write(image(), 1)
        dst.build_overviews([2, 4], Resampling.average)
    if v1:
        (folder / f"{NAME}_METADATA.json").write_text(json.dumps(V1))
    else:
        (folder / f"{NAME}.stac.v2.json").write_text(json.dumps(STAC))
    return folder
