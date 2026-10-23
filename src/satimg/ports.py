"""Commercial ports and small-boat harbors, for telling ships alongside from ships at sea.

``satimg/data/ports.json.gz``, built by ``scripts/build_ports.py``: the planet's
OpenStreetMap port areas (``industrial=port``, ``landuse=harbour``, ``harbour=yes``,
``seamark:type=harbour``), each a commercial ``"port"`` -- widened by 50 m to take in
the ships at berth -- or a ``"harbor"``: a marina or other small-boat harbor.
"""

from __future__ import annotations

import numpy
import shapely

from satimg.mask import Mask, bundled, inside

PORT = "port"
HARBOR = "harbor"


class PortMask(Mask):
    """Commercial ports."""

    kind = PORT

    def __init__(self):
        features, self._tree = bundled("ports")
        self._areas = numpy.array([f["properties"]["kind"] == self.kind for f in features], dtype=bool)

    def _contains(self, lat, lon) -> numpy.ndarray:
        return inside(self._tree, lat, lon, self._areas)


class HarborMask(PortMask):
    """Marinas and other small-boat harbors."""

    kind = HARBOR


def features_in_bbox(min_lon, min_lat, max_lon, max_lat) -> list[dict]:
    """The GeoJSON features of the ports and harbors intersecting the box, e.g. for a map."""
    features, tree = bundled("ports")
    return [features[i] for i in sorted(tree.query(shapely.box(min_lon, min_lat, max_lon, max_lat),
                                                   predicate="intersects"))]
