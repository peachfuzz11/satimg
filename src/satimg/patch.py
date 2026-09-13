"""The :class:`Patch` returned by :meth:`~satimg.product.Product.patches` /
:meth:`~satimg.product.Product.patches_at`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy
import xarray

from satimg.geometry import Window
from satimg.metadata import PatchMeta

if TYPE_CHECKING:  # pragma: no cover
    import PIL.Image


@dataclass(frozen=True, slots=True)
class Patch:
    """One window of a product's raw/visual/metadata, already sliced to that
    window -- still lazy (only ``.values`` on ``raw``/``visual`` triggers real
    computation). Built by :meth:`~satimg.product.Product._read_patch`,
    yielded by :meth:`~satimg.product.Product.patches` /
    :meth:`~satimg.product.Product.patches_at`::

        for p in product.patches_at(points, 512):
            p.raw                 # (band, y, x) DataArray for this window, band-labelled
            p.raw.sel(band="red") # ("B04" for Sentinel-2, "VV" for Sentinel-1)
            p.visual              # uint8 DataArray, same window
            p.meta.sample()       # {field: value} per-pixel angles at the patch centre
    """

    window: Window
    raw: xarray.DataArray
    visual: xarray.DataArray
    meta: PatchMeta

    # -- index passthrough ---------------------------------------
    @property
    def col(self) -> int:
        return self.window.col

    @property
    def row(self) -> int:
        return self.window.row

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        return self.window.bounds

    @property
    def center(self) -> tuple[float, float]:
        """``(col, row)`` of the patch centre in full-image pixels."""
        return self.window.center

    def image(self) -> "PIL.Image.Image":
        """Render :attr:`visual` as a PIL image."""
        import PIL.Image

        arr = self.visual
        if "band" in arr.dims:
            arr = arr.transpose("y", "x", "band")
            if arr.sizes["band"] == 1:
                arr = arr.isel(band=0)
        return PIL.Image.fromarray(numpy.asarray(arr.data))
