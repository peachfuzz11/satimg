"""A lazy, band-first view over image data plus windowed iteration.

:class:`Raster` wraps an :class:`xarray.DataArray` normalised to dims
``(band, y, x)`` and hands out :class:`Patch` objects -- a window bound to the
data it covers, so ``for patch in raster.patches(512): ...`` reads the image in
tiles without ever losing track of where each tile came from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import numpy
import xarray

from satimg.geometry import EdgeMode, Grid, Window

if TYPE_CHECKING:  # pragma: no cover
    import PIL.Image

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


class Raster:
    """Lazy ``(band, y, x)`` image data with windowed access.

    Nothing is read from disk until a patch's ``.values`` (or ``.array.compute()``)
    is requested.
    """

    def __init__(
        self,
        data: xarray.DataArray,
        *,
        transform: "Transformer | None" = None,
        name: str = "",
    ):
        self._data = as_band_yx(data)
        self._transform = transform
        self.name = name or str(data.name or "")

    # -- shape --------------------------------------------------------
    @property
    def array(self) -> xarray.DataArray:
        return self._data

    @property
    def width(self) -> int:
        return int(self._data.sizes["x"])

    @property
    def height(self) -> int:
        return int(self._data.sizes["y"])

    @property
    def bands(self) -> int:
        return int(self._data.sizes["band"])

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.bands, self.height, self.width

    @property
    def dtype(self) -> numpy.dtype:
        return self._data.dtype

    @property
    def transform(self) -> "Transformer":
        if self._transform is None:
            raise AttributeError(f"raster {self.name!r} has no transform attached")
        return self._transform

    def __repr__(self) -> str:
        return (
            f"Raster(name={self.name!r}, bands={self.bands}, "
            f"height={self.height}, width={self.width}, dtype={self.dtype})"
        )

    # -- reading ----------------------------------------------------
    def read(self, window: Window) -> xarray.DataArray:
        """Lazy sub-array for ``window``.

        The result always has the window's exact ``width``/``height`` -- if the
        window overhangs the image it is zero-padded. ``col``/``row`` are stored
        in ``.attrs`` so the origin survives a round-trip through numpy.
        """
        inner = window.clip(self.width, self.height)
        sub = self._data.isel(**inner.isel())
        pad_x = (inner.col - window.col, window.col_end - inner.col_end)
        pad_y = (inner.row - window.row, window.row_end - inner.row_end)
        if any(pad_x) or any(pad_y):
            sub = sub.pad(x=pad_x, y=pad_y, constant_values=0)
        return sub.assign_attrs(col=window.col, row=window.row)

    def values(self, window: Window) -> numpy.ndarray:
        """Materialised ``(band, y, x)`` array for ``window``."""
        return numpy.asarray(self.read(window).data)

    # -- iteration --------------------------------------------------
    def grid(
        self,
        size: int | tuple[int, int],
        *,
        overlap: int | tuple[int, int] = 0,
        edge: EdgeMode = "trim",
    ) -> Grid:
        return Grid(self.width, self.height, size, overlap, edge)

    def patches(
        self,
        size: int | tuple[int, int] = 512,
        *,
        overlap: int | tuple[int, int] = 0,
        edge: EdgeMode = "trim",
        batch: int | None = None,
    ) -> Iterator["Patch"] | Iterator[list["Patch"]]:
        """Iterate the raster in windows of ``size``.

        With ``batch=n`` the patches arrive in lists of up to ``n`` instead of one
        at a time.
        """
        grid = self.grid(size, overlap=overlap, edge=edge)
        if batch is None:
            for win in grid:
                yield Patch(self, win)
        else:
            for group in grid.batched(batch):
                yield [Patch(self, win) for win in group]

    def __iter__(self) -> Iterator["Patch"]:
        return self.patches()

    # -- lazy transforms (return new Rasters) ----------------------
    def map(self, func, *, name: str = "", keep_attrs: bool = True) -> "Raster":
        """Apply ``func`` to the underlying DataArray, returning a new Raster."""
        out = func(self._data)
        if keep_attrs:
            out = out.assign_attrs(self._data.attrs)
        return Raster(out, transform=self._transform, name=name or self.name)

    def chunk(self, size: int | tuple[int, int]) -> "Raster":
        w, h = (size, size) if isinstance(size, int) else size
        return self.map(lambda d: d.chunk({"x": w, "y": h, "band": -1}), name=self.name)

    def persist(self) -> "Raster":
        return self.map(lambda d: d.persist(), name=self.name)

    def hwc(self) -> xarray.DataArray:
        """The data as ``(y, x, band)``, or ``(y, x)`` if single-band -- ready for
        :func:`PIL.Image.fromarray`."""
        out = self._data.transpose("y", "x", "band")
        return out.isel(band=0) if out.sizes["band"] == 1 else out


class Patch:
    """A :class:`Window` bound to the raster it was cut from.

    Cheap to create and pass around; reads happen only when ``.array`` is computed
    or ``.values`` is accessed.
    """

    __slots__ = ("raster", "window")

    def __init__(self, raster: Raster, window: Window):
        self.raster = raster
        self.window = window

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

    # -- data ---------------------------------------------------
    @property
    def array(self) -> xarray.DataArray:
        """Lazy ``(band, y, x)`` sub-array."""
        return self.raster.read(self.window)

    @property
    def values(self) -> numpy.ndarray:
        """Materialised ``(band, y, x)`` array."""
        return self.raster.values(self.window)

    def image(self) -> "PIL.Image.Image":
        """Render as a PIL image (only meaningful for ``uint8`` rasters)."""
        import PIL.Image

        arr = self.array.transpose("y", "x", "band")
        arr = arr.isel(band=0) if arr.sizes["band"] == 1 else arr
        return PIL.Image.fromarray(numpy.asarray(arr.data))

    # -- geo --------------------------------------------------
    @property
    def center(self) -> tuple[float, float]:
        """``(col, row)`` of the patch centre in full-image pixels."""
        return self.window.center

    @property
    def center_latlon(self) -> tuple[float, float]:
        col, row = self.window.center
        latlon = self.raster.transform.rowcol_to_latlon((row, col))
        return float(latlon[0, 0]), float(latlon[0, 1])

    @property
    def latlon_bounds(self) -> tuple[float, float, float, float]:
        """``(min_lon, min_lat, max_lon, max_lat)`` of the patch corners."""
        c0, r0, c1, r1 = self.window.bounds
        corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
        latlon = self.raster.transform.rowcol_to_latlon(corners)
        lats, lons = latlon[:, 0], latlon[:, 1]
        return float(lons.min()), float(lats.min()), float(lons.max()), float(lats.max())

    def __repr__(self) -> str:
        return f"Patch({self.raster.name!r}, {self.window})"
