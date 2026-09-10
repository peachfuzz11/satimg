"""Small filesystem / raster IO helpers shared by the product readers."""

from __future__ import annotations

import os
from typing import Iterable

import rioxarray  # noqa: F401  (registers the rasterio xarray engine)
import xarray


def find_file(directory: str, filename: str) -> str | None:
    """Return the full path of the first ``filename`` found under ``directory``."""
    for root, _dirs, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None


def open_band(path: str, *, chunk: int = 512) -> xarray.DataArray:
    """Open one raster file lazily as a ``(band, y, x)`` DataArray.

    dask-chunked into ``chunk``-square tiles (all bands in one chunk), so a
    windowed ``isel`` decodes only the tiles it overlaps -- a bare ``.chunk()``
    makes the whole band one chunk and every window a full-band decode
    (~500 ms/window on a Sentinel scene). ``chunk`` is a starting granularity;
    :meth:`~satimg.product.Product.patches` rechunks to the loop's patch size.

    The returned array keeps a live ``_close``; the opener owns closing it.
    """
    return rioxarray.open_rasterio(
        path, chunks={"band": -1, "x": chunk, "y": chunk}, lock=False
    )


def merge_bands(
    paths: Iterable[str], *, match: int | None = None, chunk: int = 512
) -> tuple[xarray.DataArray, list[xarray.DataArray]]:
    """Open each path with :func:`open_band` and stack them along ``band``.

    Returns ``(merged, opened)`` -- ``opened`` is every per-band array, handed
    back for the caller to close. With ``match`` (an index into ``paths``) the
    other bands are nearest-neighbour reindexed onto that band's grid first.
    """
    opened = [open_band(p, chunk=chunk) for p in paths]
    if match is not None:
        target = opened[match]
        arrays = [a.reindex_like(target, method="nearest") for a in opened]
    else:
        arrays = opened
    merged = arrays[0] if len(arrays) == 1 else xarray.concat(arrays, dim="band")
    return merged, opened
