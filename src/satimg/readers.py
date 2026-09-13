"""Small filesystem / raster IO helpers shared by the product readers."""

from __future__ import annotations

import os
from typing import Iterable

import rioxarray
import xarray

#: dask chunk size (in pixels) each band is opened with, all bands in one
#: chunk. Matches the default patch size used throughout :mod:`satimg.tiling`
#: / :mod:`satimg.product`, so a loop that never overrides the default patch
#: size rechunks for free; any other size still only touches the handful of
#: :data:`_TILE` tiles a window overlaps, never the whole band.
_TILE = 512


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


def merge_bands(
    paths: Iterable[str], *, match: int | None = None
) -> xarray.DataArray:
    """Open each path lazily and stack them along ``band``.

    With ``match`` (an index into ``paths``) the other bands are
    nearest-neighbour reindexed onto that band's grid first. Each band is
    opened dask-chunked into :data:`_TILE`-square tiles -- a bare ``.chunk()``
    with no size hint collapses each band to a *single* whole-band chunk, and
    once a band is one chunk, no later rechunk (e.g.
    :meth:`~satimg.product.Product._align_view_chunks`, which resizes to the
    loop's window size) can split it into tiles without depending on a task
    that reads the entire band; every windowed patch read would then re-decode
    the whole scene. Chunking here instead means that later rechunk only ever
    depends on the handful of tiles a window overlaps.

    ``concat`` drops the per-band ``_close``, so it is wired back on: the
    returned array's ``close()`` shuts every band it opened.
    """
    opened = [
        rioxarray.open_rasterio(p, chunks={"band": -1, "x": _TILE, "y": _TILE}, lock=False)
        for p in paths
    ]
    bands = opened
    if match is not None:
        target = bands[match]
        bands = [b.reindex_like(target, method="nearest") for b in bands]
    merged = bands[0] if len(bands) == 1 else xarray.concat(bands, dim="band")
    merged.set_close(lambda: [band.close() for band in opened])
    return merged
