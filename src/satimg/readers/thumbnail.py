"""Product thumbnails, all one size."""

from __future__ import annotations

from typing import TYPE_CHECKING

import PIL.Image

if TYPE_CHECKING:  # pragma: no cover
    from satimg.source import Source

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
