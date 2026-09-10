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
    """Open one raster file lazily as a ``(band, y, x)`` DataArray.

    No dask chunking here -- a windowed ``isel`` reads straight through
    ``rasterio``. :meth:`~satimg.product.Product.patches` chunks the view to the
    loop's window size when iteration starts. The array keeps a live ``_close``,
    so ``da.close()`` shuts the GDAL dataset.
    """
    return rioxarray.open_rasterio(path)


def keep_open(view: xarray.DataArray, source: xarray.DataArray) -> xarray.DataArray:
    """Carry ``source``'s ``_close`` onto ``view`` and return it.

    ``transpose`` / ``assign_coords`` / ``astype`` / ``chunk`` all return a fresh
    DataArray without the ``_close`` hook, so a product's final ``raw`` /
    ``visual`` -- several such ops past :func:`open_band` / :func:`merge_bands` --
    would otherwise not shut its rasters on ``close()``.
    """
    view.set_close(source._close)
    return view


def merge_bands(
    paths: Iterable[str], *, match: int | None = None
) -> xarray.DataArray:
    """Open each path with :func:`open_band` and stack them along ``band``.

    With ``match`` (an index into ``paths``) the other bands are
    nearest-neighbour reindexed onto that band's grid first. Each band is given
    a single dask chunk so ``xarray.concat`` stays lazy (concatenating plain
    arrays would read every scene into memory); the real tiling happens later,
    in ``patches``. ``concat`` drops the per-band ``_close``, so it is wired
    back on -- the returned array's ``close()`` shuts every band it opened.
    """
    opened = [open_band(p) for p in paths]
    bands = [b.chunk() for b in opened]  # one chunk each -> concat stays lazy
    if match is not None:
        target = bands[match]
        bands = [b.reindex_like(target, method="nearest") for b in bands]
    merged = bands[0] if len(bands) == 1 else xarray.concat(bands, dim="band")
    merged.set_close(lambda: [band.close() for band in opened])
    return merged
