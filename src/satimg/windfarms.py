"""Offshore wind farms, for telling turbines from ships.

``satimg/data/windfarms.json.gz``, built by ``scripts/build_windfarms.py``: EMODnet's
farms (Europe) with the year they were built, and outside them OSM's offshore farm
outlines and 50 m discs around the turbines OSM maps as points, without a year.
"""

from __future__ import annotations

import numpy

from satimg.mask import Mask, bundled, inside


class WindfarmMask(Mask):
    def __init__(self, year: int | None = None):
        """The farms built by ``year`` -- every farm when it is None. A farm without a
        build year is assumed to be standing already."""
        features, self._tree = bundled("windfarms")
        self._areas = None if year is None else numpy.array(
            [(f["properties"]["year"] or 0) <= year for f in features], dtype=bool)

    def _contains(self, lat, lon) -> numpy.ndarray:
        return inside(self._tree, lat, lon, self._areas)
