"""Point masks over geographic data: every mask answers ``mask.contains(lat, lon)``."""

from __future__ import annotations

import abc
import gzip
import json
from functools import cache
from importlib.resources import files

import numpy
import shapely


class Mask(abc.ABC):
    def contains(self, lat, lon) -> numpy.ndarray:
        """Element-wise test of matching ``lat``/``lon`` points -- scalars or arrays
        of any shape -- giving booleans of that shape."""
        lat = numpy.asarray(lat, dtype=float)
        lon = numpy.asarray(lon, dtype=float)
        return self._contains(lat.ravel(), lon.ravel()).reshape(lat.shape)

    @abc.abstractmethod
    def _contains(self, lat: numpy.ndarray, lon: numpy.ndarray) -> numpy.ndarray:
        """The test over flat ``lat``/``lon`` arrays."""


@cache
def bundled(name) -> tuple[list[dict], shapely.STRtree]:
    """The GeoJSON features of ``satimg/data/<name>.json.gz`` and a tree over their
    geometries, loaded once per process."""
    with gzip.open(files("satimg") / "data" / f"{name}.json.gz") as f:
        features = json.load(f)["features"]
    return features, shapely.STRtree([shapely.geometry.shape(f["geometry"]) for f in features])


def inside(tree: shapely.STRtree, lat, lon, areas: numpy.ndarray | None = None) -> numpy.ndarray:
    """Whether each point lies in one of the tree's geometries -- of those ``areas``
    selects, when given a boolean per geometry."""
    point_idx, area_idx = tree.query(shapely.points(lon, lat), predicate="within")
    if areas is not None:
        point_idx = point_idx[areas[area_idx]]
    mask = numpy.zeros(len(lat), dtype=bool)
    mask[point_idx] = True
    return mask
