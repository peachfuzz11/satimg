"""One per-pixel quantity on its coarse grid."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import dask.array
import numpy
import xarray

from satimg.metadata.grids import bilinear
from satimg.readers.raster import CHUNK_PX
from satimg.tiling import read_window
from satimg.transform.transformer_utils import as_n2

if TYPE_CHECKING:  # pragma: no cover
    from satimg.geometry import Window

#: keys of the dict returned by every ``corners()``, in the order they map to
#: :func:`corner_center`.
CORNER_KEYS = ("top_left", "top_right", "bottom_left", "bottom_right", "center")


def corner_center(height: float, width: float) -> list[tuple[float, float]]:
    """The five ``(row, col)`` points for a ``height`` x ``width`` grid: the four
    corner pixels, then the centre (matching :meth:`~satimg.metadata.patch_meta.PatchMeta.sample`)."""
    h, w = height - 1, width - 1
    return [(0.0, 0.0), (0.0, w), (h, 0.0), (h, w), (height / 2, width / 2)]


class Field:
    """One scalar quantity sampled on a coarse ``(rows, cols, values)`` grid given
    in the product's own pixel coordinates."""

    def __init__(
        self,
        rows,
        cols,
        values,
        *,
        name: str,
        units: str = "",
        shape=None,
    ):
        self._rows = numpy.asarray(rows, dtype=float)
        self._cols = numpy.asarray(cols, dtype=float)
        self._values = numpy.asarray(values, dtype=float)
        if self._values.shape != (len(self._rows), len(self._cols)):
            raise ValueError(
                f"values {self._values.shape} do not match grid "
                f"{(len(self._rows), len(self._cols))}"
            )
        self.name = name
        self.units = units
        # (height, width) of the full product grid -- a plain tuple, or a
        # zero-arg callable resolved lazily (see _resolve_shape), so building
        # a Field never has to eagerly force a product's raw/visual open just
        # to learn its shape unless .grid/.corners() actually need it.
        self._shape = shape

    def _resolve_shape(self) -> tuple[int, int] | None:
        if self._shape is None:
            return None
        return self._shape() if callable(self._shape) else self._shape

    # -- point lookup ---------------------------------------------------
    def at(self, coords):
        """Value(s) at ``coords`` -- a ``(row, col)`` pair, a list of pairs, or an
        ``(N, 2)`` array. A single pair returns a ``float``; many return ``(N,)``.
        """
        rc = as_n2(coords)
        out = numpy.asarray(bilinear(self._rows, self._cols, self._values, rc[:, 0], rc[:, 1]))
        return float(out[0]) if numpy.ndim(coords) == 1 else out

    def corners(self) -> dict:
        """``{"top_left": v, "top_right": v, "bottom_left": v, "bottom_right": v,
        "center": v}`` -- the field at the four corners and centre of the full
        product grid."""
        shape = self._resolve_shape()
        if shape is None:
            raise AttributeError(f"field {self.name!r} has no product grid attached")
        vals = self.at(corner_center(*shape))
        return {k: float(v) for k, v in zip(CORNER_KEYS, vals)}

    # -- full grid ----------------------------------------------------
    @cached_property
    def grid(self) -> xarray.DataArray:
        """The field bilinearly upsampled to the whole product grid, lazy
        ``(y, x)``."""
        shape = self._resolve_shape()
        if shape is None:
            raise AttributeError(f"field {self.name!r} has no product grid attached")
        height, width = shape
        rows, cols, values = self._rows, self._cols, self._values
        block = dask.array.blockwise(
            lambda br, bc: bilinear(rows, cols, values, br[:, None], bc[None, :]),
            "yx",
            dask.array.arange(height, chunks=CHUNK_PX),
            "y",
            dask.array.arange(width, chunks=CHUNK_PX),
            "x",
            dtype=float,
        )
        return xarray.DataArray(block, dims=("y", "x"))

    def read(self, window: Window) -> xarray.DataArray:
        """Lazy ``(y, x)`` sub-array for ``window``."""
        return read_window(self.grid, window)

    def __repr__(self) -> str:
        u = f", units={self.units!r}" if self.units else ""
        return (
            f"Field(name={self.name!r}{u}, "
            f"grid={len(self._rows)}x{len(self._cols)})"
        )
