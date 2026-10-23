"""Offshore oil and gas platforms, for telling them from ships.

``satimg/data/platforms.json.gz``, built by ``scripts/build_platforms.py``: the planet's
OpenStreetMap offshore platforms (``man_made=offshore_platform`` or
``seamark:type=platform``). A platform mapped as a point is a 200 m disc around it.
"""

from __future__ import annotations

import numpy

from satimg.mask import Mask, bundled, inside


class PlatformMask(Mask):
    def __init__(self):
        _, self._tree = bundled("platforms")

    def _contains(self, lat, lon) -> numpy.ndarray:
        return inside(self._tree, lat, lon)
