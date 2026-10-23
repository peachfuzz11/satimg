"""Land, by :mod:`roaring_landmask`."""

from __future__ import annotations

from functools import cache

import numpy

from satimg.mask import Mask


@cache
def _roaring():
    from roaring_landmask.roaring_landmask import RoaringLandmask

    return RoaringLandmask.new()


class LandMask(Mask):
    def _contains(self, lat, lon) -> numpy.ndarray:
        return numpy.asarray(_roaring().contains_many(lon, lat), dtype=bool)
