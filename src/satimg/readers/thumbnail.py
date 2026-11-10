"""Product thumbnails, all one size."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy
import PIL.Image
import rasterio
import xarray

if TYPE_CHECKING:  # pragma: no cover
    from satimg.source.source import Source

#: uniform size every :func:`load_thumbnail` returns, regardless of what the
#: shipped quick-look's own dimensions are.
THUMBNAIL_SIZE = (200, 200)


def load_thumbnail(
    source: Source, relpath: str, *, greyscale: bool = False
) -> PIL.Image.Image:
    """The quick-look at ``relpath`` (via ``source``, so this works
    zip-native too), converted to 8-bit greyscale or RGB and resized to
    exactly :data:`THUMBNAIL_SIZE` -- an aspect-preserving cover-then-centre-crop,
    so a non-square source is cropped rather than squashed.

    ``source.open(relpath)`` is only guaranteed to stay live for the
    ``with`` block's duration (a zip member has no independent handle once
    the ``ZipFile`` moves on), so the image is decoded (``.load()``) before
    that block exits.
    """
    with source.open(relpath) as f:
        img = PIL.Image.open(f)
        img.load()
    return fit_thumbnail(img.convert("L" if greyscale else "RGB"))


def fit_thumbnail(img: PIL.Image.Image) -> PIL.Image.Image:
    """``img`` resized to exactly :data:`THUMBNAIL_SIZE` -- covered, then
    centre-cropped (see :func:`load_thumbnail`)."""
    target_w, target_h = THUMBNAIL_SIZE
    scale = max(target_w / img.width, target_h / img.height)
    covered = img.resize(
        (round(img.width * scale), round(img.height * scale)),
        PIL.Image.Resampling.LANCZOS,
    )
    left = (covered.width - target_w) // 2
    top = (covered.height - target_h) // 2
    return covered.crop((left, top, left + target_w, top + target_h))


def thumbnail_from_visual(visual: xarray.DataArray) -> PIL.Image.Image:
    """A thumbnail rendered from a ``(band, y, x)`` ``uint8`` visual (one
    greyscale or three RGB bands) -- for products that ship no quick-look.
    Reads it decimated to ~400 pixels on the long side, then fits it to
    :data:`THUMBNAIL_SIZE` (see :func:`fit_thumbnail`)."""
    step = max(1, max(visual.sizes["y"], visual.sizes["x"]) // 400)
    pixels = visual[:, ::step, ::step].values
    img = pixels[0] if pixels.shape[0] == 1 else numpy.moveaxis(pixels, 0, -1)
    return fit_thumbnail(PIL.Image.fromarray(img))


def render_thumbnail(
    path: str, render: Callable[[xarray.DataArray, tuple[int, int]], xarray.DataArray]
) -> PIL.Image.Image:
    """A greyscale thumbnail of the first band of the raster at ``path``,
    read decimated to about :data:`THUMBNAIL_SIZE` -- off its overviews, for
    a cloud-optimised GeoTIFF, so only a few hundred kilobytes are read --
    and turned into ``uint8`` by ``render`` (a product's ``_render_visual``),
    for a product that ships no quick-look of its own."""
    with rasterio.open(path) as src:
        scale = max(src.height, src.width) / max(THUMBNAIL_SIZE)
        shape = (max(1, round(src.height / scale)), max(1, round(src.width / scale)))
        data = src.read(1, out_shape=shape)
    raw = xarray.DataArray(data[numpy.newaxis], dims=("band", "y", "x"))
    return fit_thumbnail(PIL.Image.fromarray(render(raw, shape).values[0], mode="L"))
