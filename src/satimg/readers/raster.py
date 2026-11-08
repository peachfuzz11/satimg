"""The :class:`Raster` interface: one lazily read image file or dataset.

Every product's pixels -- whatever the format -- are read through it, so
they all open, chunk, decimate and close the same way. GDAL reads most
(:class:`GDALRaster`); ``h5py`` reads the HDF5 datasets GDAL cannot name
(:class:`~satimg.readers.hdf5.HDF5Raster`).
"""

from __future__ import annotations

import abc
from functools import cached_property

import rasterio
import rioxarray
import xarray

#: The one default dask chunk size (pixels) satimg falls back to wherever it
#: needs *some* chunking but has no caller-given window size to use instead:
#: :attr:`~satimg.product.Product.raw` / :attr:`~satimg.product.Product.visual`
#: opened outside a ``patches()`` loop, every ``patches_at()`` loop,
#: :attr:`~satimg.metadata.field.Field.grid`, and the fixed granularity a
#: :class:`~satimg.readers.stack.BandStack` aligns bands at before resampling
#: them onto one grid.
CHUNK_PX = 1024

Tile = int | tuple[int, int]


def tile_xy(tile: Tile) -> tuple[int, int]:
    """``tile`` as ``(width, height)``."""
    return (tile, tile) if isinstance(tile, int) else tile


def keep_open(view: xarray.DataArray, source: xarray.DataArray) -> xarray.DataArray:
    """Carry ``source``'s ``_close`` onto ``view`` and return it.

    ``transpose`` / ``assign_coords`` / ``astype`` / ``chunk`` each return a
    fresh DataArray without the ``_close`` hook, and a held reference to any of
    them keeps the raster open past a garbage collection -- so a product's final
    ``raw`` / ``visual`` would not shut its rasters on ``close()`` without this.
    """
    view.set_close(source._close)
    return view


class Raster(abc.ABC):
    """One image file or dataset, read lazily."""

    @abc.abstractmethod
    def open(self, tile: Tile, overview: int = 1) -> xarray.DataArray:
        """The raster as a lazy ``(band, y, x)`` array, dask-chunked to
        ``tile`` (``int`` or ``(width, height)``) with every band in one
        chunk -- at its ``overview``-times decimated overview unless that is
        1 (a ``ValueError`` if it has none). ``close()`` releases the file."""

    @property
    @abc.abstractmethod
    def overviews(self) -> list[int]:
        """The decimation factors :meth:`open` takes besides 1."""


class GDALRaster(Raster):
    """A raster GDAL opens by ``path`` -- a file, a ``/vsizip/`` member, an
    ``HDF5:`` dataset or an in-memory VRT (:func:`~satimg.readers.vrt.vrt`).
    ``dtype`` casts it as it reads (lazily)."""

    def __init__(self, path: str, dtype: str | None = None):
        self.path = path
        self._dtype = dtype

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.path!r})"

    @cached_property
    def overviews(self) -> list[int]:
        with rasterio.open(self.path) as src:
            return list(src.overviews(1))

    def open(self, tile: Tile, overview: int = 1) -> xarray.DataArray:
        if overview == 1:
            opened = rioxarray.open_rasterio(self.path)
        elif overview in self.overviews:
            opened = rioxarray.open_rasterio(self.path, OVERVIEW_LEVEL=self.overviews.index(overview))
        else:
            raise ValueError(f"{self.path} has no {overview}x overview, only {self.overviews}")
        tw, th = tile_xy(tile)
        # chunk before anything else: a plain DataArray's astype() computes
        # eagerly, and one whole-band chunk would make every windowed read
        # decode the entire band
        da = opened.chunk({"band": -1, "x": tw, "y": th})
        if self._dtype is not None and da.dtype != self._dtype:
            da = da.astype(self._dtype)
        return keep_open(da, opened)
