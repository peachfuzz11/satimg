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


def open_band(path: str) -> xarray.DataArray:
    """Open a single raster file lazily as a ``(band, y, x)`` DataArray."""
    return rioxarray.open_rasterio(path).chunk()


def merge_bands(paths: Iterable[str], *, match: int | None = None) -> xarray.DataArray:
    """Open every path and concatenate along ``band``.

    When the sources differ in sampling, pass ``match`` (an index into ``paths``)
    and the others are reindexed onto that band's grid with nearest-neighbour.
    """
    paths = list(paths)
    arrays = [open_band(p) for p in paths]
    if match is not None:
        target = arrays[match]
        arrays = [a.reindex_like(target, method="nearest").chunk() for a in arrays]
    return xarray.concat(arrays, dim="band")
