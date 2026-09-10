"""Small filesystem / raster IO helpers shared by the product readers."""

from __future__ import annotations

import os
from typing import Iterable

import rioxarray  # noqa: F401  (registers the rasterio xarray engine)
import xarray

#: Read granularity for :func:`open_band`. dask chunks of this size, all bands in
#: one chunk. A windowed read decodes only the tiles it overlaps; a window
#: smaller than this still costs one full tile per band (512x512 uint16 ~= 0.5 MB).
#: Not derived from the file: a GeoTIFF/JP2's own block size is typically 8192 or
#: the whole band -- far larger than the patches satimg iterates -- and an
#: untiled TIFF has none. This is the value rioxarray-style readers pick anyway.
_TILE = 512
_CHUNKS = {"band": -1, "x": _TILE, "y": _TILE}


def find_file(directory: str, filename: str) -> str | None:
    """Return the full path of the first ``filename`` found under ``directory``."""
    for root, _dirs, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None


def open_band(path: str, *, sources: list | None = None) -> xarray.DataArray:
    """Open a single raster file lazily as a ``(band, y, x)`` DataArray.

    Opened dask-backed with :data:`_CHUNKS`, so a windowed ``isel`` decodes only
    the tiles it overlaps. A bare ``.chunk()`` instead makes the whole band one
    chunk, turning every window read into a full-band decode (~500 ms/window on
    a Sentinel scene vs a few ms here).

    ``rioxarray`` still attaches a ``_close`` hook to the returned array; pass a
    ``sources`` list and the opener is appended to it so the owning
    :class:`~satimg.product.Product` can shut the GDAL dataset on ``close()``.
    """
    src = rioxarray.open_rasterio(path, chunks=_CHUNKS, lock=False)
    if sources is not None:
        sources.append(src)
    return src


def merge_bands(
    paths: Iterable[str], *, match: int | None = None, sources: list | None = None
) -> xarray.DataArray:
    """Open every path and concatenate along ``band``.

    When the sources differ in sampling, pass ``match`` (an index into ``paths``)
    and the others are reindexed onto that band's grid with nearest-neighbour.
    The reindexed arrays keep the block chunking :func:`open_band` gave them --
    no trailing ``.chunk()`` to collapse them back to one chunk per band.

    ``sources`` is forwarded to :func:`open_band`; every underlying opener is
    appended to it for the caller to close.
    """
    paths = list(paths)
    arrays = [open_band(p, sources=sources) for p in paths]
    if match is not None:
        target = arrays[match]
        arrays = [a.reindex_like(target, method="nearest") for a in arrays]
    return xarray.concat(arrays, dim="band")
