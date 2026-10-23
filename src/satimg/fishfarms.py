"""Offshore fish farms, for telling their net cages and feed barges from ships.

``satimg/data/fishfarms.json.gz``, built by ``scripts/build_fishfarms.py``: EMODnet's
finfish farms at sea (Europe) and OpenStreetMap's offshore fish farms. A farm mapped as
a point is a 250 m disc around it.
"""

from __future__ import annotations

import numpy

from satimg.mask import Mask, bundled, inside


class FishFarmMask(Mask):
    def __init__(self):
        _, self._tree = bundled("fishfarms")

    def _contains(self, lat, lon) -> numpy.ndarray:
        return inside(self._tree, lat, lon)
