"""Windowed iteration over lazy image data.

Free functions plus :class:`Patch` that turn an :class:`xarray.DataArray` --
normalised to ``(band, y, x)`` by :func:`as_band_yx`, or a bare ``(y, x)``
metadata grid -- into a stream of :class:`Patch` objects: a :class:`Window` bound
to the data it covers, so ``for p in patches(da, 512): ...`` reads the array in
tiles without ever losing track of where each tile came from.
"""

from __future__ import annotations

from itertools import batched
from typing import TYPE_CHECKING, Iterable, Iterator

import numpy
import xarray

from satimg.geometry import EdgeMode, Grid, Window, windows_at

if TYPE_CHECKING:  # pragma: no cover
    import PIL.Image

    from satimg.product import Product
    from satimg.transform import Transformer


def as_band_yx(data: xarray.DataArray) -> xarray.DataArray:
    """Return ``data`` with dims exactly ``(band, y, x)``, adding a singleton
    ``band`` axis if the array is 2-D."""
    if "band" not in data.dims:
        data = data.expand_dims("band")
    missing = {"band", "y", "x"} - set(data.dims)
    if missing:
        raise ValueError(f"expected band/y/x dims, missing {missing}: got {data.dims}")
    return data.transpose("band", "y", "x", *[d for d in data.dims if d not in ("band", "y", "x")])


def label_bands(da: xarray.DataArray, names) -> xarray.DataArray:
    """Attach ``names`` as the ``band`` coordinate of ``da`` (a ``(band, y, x)``
    array). ``len(names)`` must equal the band count."""
    names = list(names)
    if len(names) != da.sizes["band"]:
        raise ValueError(
            f"got {len(names)} band names for {da.sizes['band']} bands: {names}"
        )
    return da.assign_coords(band=names)


def read_window(da: xarray.DataArray, window: Window) -> xarray.DataArray:
    """Lazy sub-array of ``da`` for ``window``.

    The result always has the window's exact ``width``/``height`` -- if the
    window overhangs the array it is zero-padded. ``col``/``row`` are stored in
    ``.attrs`` so the origin survives a round-trip through numpy. Only the ``x``
    and ``y`` axes are touched, so this works for ``(band, y, x)`` and bare
    ``(y, x)`` arrays alike.
    """
    w, h = int(da.sizes["x"]), int(da.sizes["y"])
    inner = window.clip(w, h)
    sub = da.isel(**inner.isel())
    pad_x = (inner.col - window.col, window.col_end - inner.col_end)
    pad_y = (inner.row - window.row, window.row_end - inner.row_end)
    if any(pad_x) or any(pad_y):
        sub = sub.pad(x=pad_x, y=pad_y, constant_values=0)
    return sub.assign_attrs(col=window.col, row=window.row)


def patches(
    da: xarray.DataArray,
    size: int | tuple[int, int] = 512,
    *,
    overlap: int | tuple[int, int] = 0,
    edge: EdgeMode = "pad",
    batch: int | None = None,
    transformer: "Transformer | None" = None,
    product: "Product | None" = None,
) -> Iterator["Patch"] | Iterator[list["Patch"]]:
    """Iterate ``da`` in windows of ``size``.

    ``edge`` defaults to ``"pad"`` so every patch is exactly ``size`` (the border
    overhang is zero-filled); pass ``edge="trim"`` for lossless re-assembly or
    ``"skip"`` to keep only full interior tiles. With ``batch=n`` the patches
    arrive in lists of up to ``n`` instead of one at a time.

    ``transformer`` lets the yielded patches resolve ``.center_latlon``;
    ``product`` binds ``.meta``. :meth:`~satimg.product.Product.patches` passes
    both.
    """
    grid = Grid(int(da.sizes["x"]), int(da.sizes["y"]), size, overlap, edge)
    if batch is None:
        for win in grid:
            yield Patch(da, win, transformer, product)
    else:
        for group in grid.batched(batch):
            yield [Patch(da, win, transformer, product) for win in group]


def patches_at(
    da: xarray.DataArray,
    points: Iterable[tuple[float, float]],
    size: int | tuple[int, int] = 512,
    *,
    batch: int | None = None,
    transformer: "Transformer | None" = None,
    product: "Product | None" = None,
) -> Iterator["Patch"] | Iterator[list["Patch"]]:
    """Iterate patches centred on caller-supplied ``(col, row)`` pixel points.

    Where :func:`patches` sweeps a regular grid over the whole array, this visits
    only ``points`` -- one patch per point, in order. Each patch is exactly
    ``size``; any part lying outside the array is zero-filled, so a point near
    (or past) an edge still yields a full-size patch. With ``batch=n`` the
    patches arrive in lists of up to ``n``.
    """
    windows = windows_at(points, size)
    if batch is None:
        for win in windows:
            yield Patch(da, win, transformer, product)
    else:
        for group in batched(windows, batch):
            yield [Patch(da, win, transformer, product) for win in group]


class Patch:
    """A :class:`Window` bound to the :class:`xarray.DataArray` it was cut from.

    Cheap to create and pass around; reads happen only when ``.array`` is
    computed or ``.values`` is accessed.
    """

    __slots__ = ("data", "window", "_transformer", "_product")

    def __init__(
        self,
        data: xarray.DataArray,
        window: Window,
        transformer: "Transformer | None" = None,
        product: "Product | None" = None,
    ):
        self.data = data
        self.window = window
        self._transformer = transformer
        self._product = product

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

    # -- data ---------------------------------------------------
    @property
    def array(self) -> xarray.DataArray:
        """Lazy sub-array for this window (same dims as the source)."""
        return read_window(self.data, self.window)

    @property
    def values(self) -> numpy.ndarray:
        """Materialised array for this window (shorthand for ``.raw.values``)."""
        return numpy.asarray(self.array.data)

    @property
    def raw(self) -> xarray.DataArray:
        """This window read from ``product.raw`` (band-labelled). Only available
        on patches from ``product.patches()`` / ``product.patches_at()``."""
        return read_window(self._require_product().raw, self.window)

    @property
    def visual(self) -> xarray.DataArray:
        """This window read from ``product.visual`` (uint8, band-labelled). Only
        available on patches from ``product.patches()`` /
        ``product.patches_at()``."""
        product = self._require_product()
        visual = product.visual
        if (visual.sizes["x"], visual.sizes["y"]) != (
            product.raw.sizes["x"],
            product.raw.sizes["y"],
        ):
            raise ValueError(
                "product.visual is not co-registered with product.raw; "
                "patch.visual needs matching grids"
            )
        return read_window(visual, self.window)

    def _require_product(self) -> "Product":
        if self._product is None:
            raise AttributeError(
                "patch has no product -- iterate product.patches() / "
                "product.patches_at() to use .raw / .visual / .meta"
            )
        return self._product

    @property
    def meta(self) -> "PatchMeta":
        """Lazy per-pixel metadata for this window (see
        :class:`~satimg.metadata.PatchMeta`). Only available on patches from
        ``product.patches()`` / ``iter(product)``."""
        from satimg.metadata import PatchMeta

        return PatchMeta(self._require_product().metadata, self.window)

    def image(self) -> "PIL.Image.Image":
        """Render as a PIL image (only meaningful for ``uint8`` data)."""
        import PIL.Image

        arr = self.array
        if "band" in arr.dims:
            arr = arr.transpose("y", "x", "band")
            if arr.sizes["band"] == 1:
                arr = arr.isel(band=0)
        return PIL.Image.fromarray(numpy.asarray(arr.data))

    # -- geo --------------------------------------------------
    def _xf(self) -> "Transformer":
        if self._transformer is None:
            raise AttributeError(
                "patch has no transformer -- iterate product.patches() or pass "
                "transformer= to satimg.patches()"
            )
        return self._transformer

    @property
    def center_latlon(self) -> tuple[float, float]:
        col, row = self.window.center
        latlon = self._xf().rowcol_to_latlon((row, col))
        return float(latlon[0, 0]), float(latlon[0, 1])

    @property
    def latlon_bounds(self) -> tuple[float, float, float, float]:
        """``(min_lon, min_lat, max_lon, max_lat)`` of the patch corners."""
        c0, r0, c1, r1 = self.window.bounds
        corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
        latlon = self._xf().rowcol_to_latlon(corners)
        lats, lons = latlon[:, 0], latlon[:, 1]
        return float(lons.min()), float(lats.min()), float(lons.max()), float(lats.max())

    def __repr__(self) -> str:
        return f"Patch({self.window})"
