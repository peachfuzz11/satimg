"""Small filesystem / raster IO helpers shared by the product readers."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Iterable

import rioxarray
import xarray

if TYPE_CHECKING:  # pragma: no cover
    import PIL.Image

    from satimg.source import Source

#: uniform size every :func:`load_thumbnail` returns, regardless of what the
#: shipped quick-look's own dimensions are.
THUMBNAIL_SIZE = (200, 200)


def load_thumbnail(
    source: "Source", relpath: str, *, greyscale: bool = False
) -> "PIL.Image.Image":
    """The quick-look at ``relpath`` (via ``source``, so this works
    zip-native too), converted to 8-bit greyscale or RGB and resized to
    exactly :data:`THUMBNAIL_SIZE` -- an aspect-preserving cover-then-centre-crop,
    so a non-square source is cropped rather than squashed.

    ``source.open(relpath)`` is only guaranteed to stay live for the
    ``with`` block's duration (a zip member has no independent handle once
    the ``ZipFile`` moves on), so the image is decoded (``.load()``) before
    that block exits.
    """
    import PIL.Image

    with source.open(relpath) as f:
        img = PIL.Image.open(f)
        img.load()
    img = img.convert("L" if greyscale else "RGB")

    target_w, target_h = THUMBNAIL_SIZE
    scale = max(target_w / img.width, target_h / img.height)
    covered = img.resize(
        (round(img.width * scale), round(img.height * scale)),
        PIL.Image.Resampling.LANCZOS,
    )
    left = (covered.width - target_w) // 2
    top = (covered.height - target_h) // 2
    return covered.crop((left, top, left + target_w, top + target_h))


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


#: The one default dask chunk size (pixels) satimg falls back to wherever it
#: needs *some* chunking but has no caller-given window size to use instead:
#: :attr:`~satimg.product.Product.raw` / :attr:`~satimg.product.Product.visual`
#: opened outside a ``patches()`` / ``patches_at()`` loop,
#: :attr:`~satimg.metadata.Field.grid`, and the fixed granularity
#: ``merge_bands`` aligns bands at internally before a ``match=`` reindex (see
#: the note in ``merge_bands`` for why that one can't just use the caller's
#: own tile).
CHUNK_PX = 1024


def merge_bands(
    paths: Iterable[str],
    *,
    match: int | None = None,
    tile: int | tuple[int, int] = CHUNK_PX,
) -> xarray.DataArray:
    """Open each path lazily and stack them along ``band``.

    With ``match`` (an index into ``paths``) the other bands are
    nearest-neighbour reindexed onto that band's grid first. Either way, the
    returned array is dask-chunked to ``tile`` (all bands in one chunk) --
    ``Product.patches()`` / ``patches_at()`` pass the loop's own window size,
    so bands read chunked to precisely what will be read.

    A bare ``.chunk()`` with no size hint would collapse each band to a
    *single* whole-band chunk. Dask can't read part of an unsplit chunk, so
    any later windowed read -- direct, or through a ``reindex_like`` onto
    another band's grid -- would decode the *entire* band first, regardless
    of any rechunk downstream (rechunking a single chunk into tiles still
    depends on the one task that reads all of it).

    With ``match=``, bands are chunked at the fixed :data:`CHUNK_PX` --
    not ``tile`` -- before the reindex: ``reindex_like``'s nearest-neighbour
    lookup builds a dask task per output chunk, so aligning directly at a
    small ``tile`` (e.g. a 64px patch loop) can turn one array into hundreds
    of thousands of chunks and take *minutes* just to build the graph, before
    any pixel is read. The reindexed, concatenated result is then rechunked
    to ``tile`` in one final step instead -- splitting an
    already-reasonably-chunked array into smaller pieces is cheap (no data
    read to redraw chunk boundaries), unlike the reindex itself. Without
    ``match=`` every band already shares one grid, so there is no such
    penalty and bands chunk straight to ``tile``.

    ``concat`` drops the per-band ``_close``, so it is wired back on: the
    returned array's ``close()`` shuts every band it opened.
    """
    tw, th = (tile, tile) if isinstance(tile, int) else tile
    opened = [rioxarray.open_rasterio(p) for p in paths]
    if match is None:
        bands = [b.chunk({"x": tw, "y": th}) for b in opened]
        merged = bands[0] if len(bands) == 1 else xarray.concat(bands, dim="band")
    else:
        # deliberately CHUNK_PX here, not tw/th: reindex_like must never see
        # the caller's own (possibly small) tile -- see the docstring.
        bands = [b.chunk({"x": CHUNK_PX, "y": CHUNK_PX}) for b in opened]
        target = bands[match]
        bands = [b.reindex_like(target, method="nearest") for b in bands]
        merged = xarray.concat(bands, dim="band")
        merged = merged.chunk({"band": -1, "x": tw, "y": th})
    merged.set_close(lambda: [band.close() for band in opened])
    return merged
