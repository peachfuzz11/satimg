"""Offshore oil and gas platforms, for telling them from ships.

The areas are the planet's OpenStreetMap offshore platforms (``man_made=offshore_platform``
or ``seamark:type=platform``), bundled as ``satimg/data/platforms.json.gz`` by
``scripts/build_platforms.py``. A platform mapped as a point is a 200 m disc around it.
"""

from __future__ import annotations

import gzip
import json
from importlib.resources import files

import numpy
import shapely


class PlatformMask:
    def __init__(self):
        with gzip.open(files("satimg") / "data" / "platforms.json.gz") as f:
            features = json.load(f)["features"]
        self._tree = shapely.STRtree([shapely.geometry.shape(f["geometry"]) for f in features])

    def contains_many(self, lon, lat) -> numpy.ndarray:
        """Element-wise test of matching ``lon``/``lat`` arrays for lying at a platform."""
        lon = numpy.asarray(lon, dtype=float)
        lat = numpy.asarray(lat, dtype=float)
        point_idx, _ = self._tree.query(shapely.points(lon.ravel(), lat.ravel()), predicate="within")
        mask = numpy.zeros(lon.size, dtype=bool)
        mask[point_idx] = True
        return mask.reshape(lon.shape)
