"""The :class:`Transformer` interface every georeferencing kind implements."""

from __future__ import annotations

import abc
from collections.abc import Sequence

import numpy

#: a ``(a, b)`` pair, a list of pairs, or an ``(N, 2)`` array
Coords = Sequence[float] | Sequence[Sequence[float]] | numpy.ndarray


def as_n2(coords: Coords) -> numpy.ndarray:
    """``coords`` as a float ``(N, 2)`` array."""
    arr = numpy.array(coords, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, 2)
    return arr


class Transformer(abc.ABC):
    """Pixel <-> geographic coordinate conversion.

    Both directions accept :data:`Coords` and return an ``(N, 2)`` array.
    ``rowcol`` is ``(row, col)`` = ``(y, x)``; ``latlon`` is ``(lat, lon)``
    in degrees (EPSG:4326).

    Every transformer round-trips through a JSON-serialisable dict
    (:meth:`to_dict`, then the subclass's ``from_dict``, or
    :func:`~satimg.transform.transformer_factory.load_transformer` for any kind), so a
    product's georeferencing can be stored and used after the product file
    is gone.
    """

    @abc.abstractmethod
    def rowcol_to_latlon(self, coords: Coords) -> numpy.ndarray:
        """``(lat, lon)`` of each ``(row, col)``."""

    @abc.abstractmethod
    def latlon_to_rowcol(self, coords: Coords) -> numpy.ndarray:
        """``(row, col)`` of each ``(lat, lon)``."""

    @abc.abstractmethod
    def to_dict(self) -> dict:
        """A JSON-serialisable dict the subclass's ``from_dict`` turns back
        into an equal transformer."""

    def north_angle(self, coords: Coords) -> numpy.ndarray:
        """The direction true north points in the image at each ``(row, col)``,
        in degrees clockwise from image "up" (decreasing row), ``[0, 360)``
        -- an ``(N,)`` array.

        ``0`` for a north-up geographic grid; a few degrees either way across
        a UTM zone (grid convergence); anything at all on a polar
        stereographic grid. Found by stepping a short way north from each
        point and seeing where that lands in pixels, so it holds for any
        transformer whose :meth:`latlon_to_rowcol` is sub-pixel (not a
        :class:`~satimg.transform.gcp_transformer.GCPTransformer`, which snaps to whole
        pixels).
        """
        rc = as_n2(coords)
        latlon = self.rowcol_to_latlon(rc)
        # ~10 m north; towards the equator instead right at a pole
        step = numpy.where(latlon[:, 0] > 89.9, -1e-4, 1e-4)
        ahead = self.latlon_to_rowcol(numpy.column_stack((latlon[:, 0] + step, latlon[:, 1])))
        d_row, d_col = (ahead - rc).T * numpy.sign(step)
        return numpy.rad2deg(numpy.arctan2(d_col, -d_row)) % 360.0
