"""Offshore wind farm areas, for telling turbines from ships.

The polygons are EMODnet's wind farms in Production, Construction or Test site
status, bundled as ``satimg/data/windfarms.json.gz``.
"""

from __future__ import annotations

import gzip
import json
from importlib.resources import files

import numpy
import shapely


class WindfarmMask:
    def __init__(self):
        with gzip.open(files("satimg") / "data" / "windfarms.json.gz") as f:
            features = json.load(f)["features"]
        self._tree = shapely.STRtree([shapely.geometry.shape(f["geometry"]) for f in features])
        # the year the farm was built; a farm without one is assumed to be standing already
        self._years = numpy.array([f["properties"]["year"] or 0 for f in features])

    def contains_many(self, lon, lat, year: int | None = None) -> numpy.ndarray:
        """Element-wise test of matching ``lon``/``lat`` arrays for lying inside a wind
        farm built by ``year`` (any farm when ``year`` is None)."""
        lon = numpy.asarray(lon, dtype=float)
        lat = numpy.asarray(lat, dtype=float)
        point_idx, farm_idx = self._tree.query(shapely.points(lon.ravel(), lat.ravel()), predicate="within")
        if year is not None:
            point_idx = point_idx[self._years[farm_idx] <= year]
        mask = numpy.zeros(lon.size, dtype=bool)
        mask[point_idx] = True
        return mask.reshape(lon.shape)
