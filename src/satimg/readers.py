"""Small filesystem / raster IO helpers shared by the product readers."""

from __future__ import annotations

import os
from typing import Iterable

import rioxarray
import xarray


def find_file(directory: str, filename: str) -> str | None:
    """Return the full path of the first ``filename`` found under ``directory``."""
    for root, _dirs, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None


def keep_open(view: xarray.DataArray, source: xarray.DataArray) -> xarray.DataArray:
    """Carry ``source``'s ``_close`` onto ``view`` and return it.

    ``transpose`` / ``assign_coords`` / ``astype`` / ``chunk`` each return a
    fresh DataArray without the ``_close`` hook, and a held reference to any of
    them keeps the raster open past a garbage collection -- so a product's final
    ``raw`` / ``visual`` would not shut its rasters on ``close()`` without this.
    """
    view.set_close(source._close)
    return view


#: dask chunk size (pixels) each band gets on open. A single whole-array chunk
#: (``.chunk()`` with no arguments) would still make ``xarray.concat`` lazy,
#: but dask can't read *part* of an unsplit chunk -- so any later windowed
#: read (directly, or through a ``reindex_like`` onto another band's grid
#: below) would decode the entire band first. Tiling it here keeps every
#: downstream read -- including :meth:`~satimg.product.Product.patches`'s
#: rechunk to the window size -- properly lazy and windowed.
CHUNK_PX = 1024


def merge_bands(
    paths: Iterable[str], *, match: int | None = None
) -> xarray.DataArray:
    """Open each path lazily and stack them along ``band``.

    With ``match`` (an index into ``paths``) the other bands are
    nearest-neighbour reindexed onto that band's grid first.

    ``concat`` drops the per-band ``_close``, so it is wired back on: the
    returned array's ``close()`` shuts every band it opened.
    """
    opened = [rioxarray.open_rasterio(p) for p in paths]
    bands = [b.chunk({"x": CHUNK_PX, "y": CHUNK_PX}) for b in opened]
    if match is not None:
        target = bands[match]
        bands = [b.reindex_like(target, method="nearest") for b in bands]
    merged = bands[0] if len(bands) == 1 else xarray.concat(bands, dim="band")
    merged.set_close(lambda: [band.close() for band in opened])
    return merged
