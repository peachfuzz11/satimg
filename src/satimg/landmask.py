"""Thin wrapper over :mod:`roaring_landmask` for point / grid land tests."""

from __future__ import annotations

import numpy


class LandMask:
    def __init__(self):
        from roaring_landmask.roaring_landmask import RoaringLandmask

        self._mask = RoaringLandmask.new()

    def contains(self, lon: float, lat: float) -> bool:
        """True if a single ``(lon, lat)`` point is on land."""
        return bool(self._mask.contains(lon, lat))

    def contains_many(self, lon, lat) -> numpy.ndarray:
        """Element-wise land test over matching ``lon``/``lat`` arrays."""
        lon = numpy.asarray(lon)
        lat = numpy.asarray(lat)
        flat = self._mask.contains_many(lon.ravel(), lat.ravel())
        return numpy.asarray(flat).reshape(lon.shape)

    def contains_grid(self, latlon: numpy.ndarray) -> numpy.ndarray:
        """Land test over an ``(..., 2)`` array of ``(lat, lon)`` -> boolean grid."""
        latlon = numpy.asarray(latlon)
        return self.contains_many(latlon[..., 1], latlon[..., 0])
