"""Port and harbor areas, for telling ships alongside from ships at sea.

The polygons are the planet's OpenStreetMap port areas (``industrial=port``,
``landuse=harbour``, ``harbour=yes``, ``seamark:type=harbour``), bundled as
``satimg/data/ports.json.gz`` by ``scripts/build_ports.py``. Each is a commercial
``"port"`` or a ``"harbor"`` -- a marina or other small-boat harbor.
"""

from __future__ import annotations

import gzip
import json
from importlib.resources import files

import numpy
import shapely

PORT = "port"
HARBOR = "harbor"


class PortMask:
    def __init__(self):
        with gzip.open(files("satimg") / "data" / "ports.json.gz") as f:
            self._features = json.load(f)["features"]
        self._tree = shapely.STRtree([shapely.geometry.shape(f["geometry"]) for f in self._features])
        self._harbor = numpy.array([f["properties"]["kind"] == HARBOR for f in self._features], dtype=bool)

    def kinds_many(self, lon, lat) -> numpy.ndarray:
        """Element-wise :data:`PORT` or :data:`HARBOR` for matching ``lon``/``lat``
        arrays, None outside both. A harbor wins over a port around it."""
        lon = numpy.asarray(lon, dtype=float)
        lat = numpy.asarray(lat, dtype=float)
        point_idx, area_idx = self._tree.query(shapely.points(lon.ravel(), lat.ravel()), predicate="within")
        kinds = numpy.full(lon.size, None, dtype=object)
        kinds[point_idx[~self._harbor[area_idx]]] = PORT
        kinds[point_idx[self._harbor[area_idx]]] = HARBOR
        return kinds.reshape(lon.shape)

    def features_in_bbox(self, min_lon, min_lat, max_lon, max_lat) -> list[dict]:
        """The GeoJSON features of the ports and harbors intersecting the box."""
        box = shapely.box(min_lon, min_lat, max_lon, max_lat)
        return [self._features[i] for i in sorted(self._tree.query(box, predicate="intersects"))]
