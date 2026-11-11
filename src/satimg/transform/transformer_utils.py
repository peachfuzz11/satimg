"""Helpers the transformers and their users share: coordinate arrays, and
the ground size of a pixel."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy

if TYPE_CHECKING:  # pragma: no cover
    from satimg.transform.transformer import Coords, Transformer


def as_n2(coords: Coords) -> numpy.ndarray:
    """``coords`` as a float ``(N, 2)`` array."""
    arr = numpy.array(coords, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, 2)
    return arr


#: mean Earth radius (m), for :func:`pixel_spacing`
_EARTH_RADIUS = 6_371_008.8


def pixel_spacing(transformer: Transformer, shape: tuple[int, int]) -> tuple[float, float]:
    """Ground size (m) of one pixel step along ``(rows, cols)`` at the centre
    of a ``shape`` grid -- for a grid whose steps are not a fixed number of
    metres (degrees, rotated, or sensor geometry)."""
    r, c = shape[0] / 2, shape[1] / 2
    (lat0, lon0), (lat_r, lon_r), (lat_c, lon_c) = transformer.rowcol_to_latlon(
        [(r, c), (r + 1, c), (r, c + 1)]
    )

    def metres(lat, lon):
        dy = numpy.deg2rad(lat - lat0)
        dx = numpy.deg2rad(lon - lon0) * numpy.cos(numpy.deg2rad(lat0))
        return float(_EARTH_RADIUS * numpy.hypot(dx, dy))

    return metres(lat_r, lon_r), metres(lat_c, lon_c)
