"""The :class:`Product` abstraction.

A product is a directory of satellite imagery + metadata. Whatever the sensor,
it offers the same two views on the pixels --

* ``product.raw``    -- every native band merged onto one grid, native dtype, lazy;
* ``product.visual`` -- the sensor's ``uint8`` visualisation, lazy: a 3-band
  true-colour image for optical sensors, a 1-band greyscale image for SAR;

-- and one way to walk them::

    for patch in product.patches(512, overlap=64):
        tile = patch.raw.values      # (band, y, x), origin at patch.col / patch.row

``for patch in product`` is shorthand for ``product.patches()``.

``patches`` / ``patches_at`` open their own raw / visual view, separate from
:attr:`Product.raw` / :attr:`Product.visual`, dask-chunked to exactly the
loop's window size -- so each patch read pulls a single tile -- and close it
when the loop ends (return, break, or an exception). Each yielded
:class:`Patch` already carries its own window's ``raw`` / ``visual`` / ``meta``
sliced out (still lazy -- only ``.values`` on ``raw``/``visual`` triggers real
computation). ``product.raw`` / ``product.visual`` accessed directly (outside
a loop) are cached on the product and released when its ``with`` block exits,
so use ``with satimg.open(path) as product`` (and :func:`satimg.open_zip`) if
you use those directly across many products, to avoid leaking handles.
"""

from __future__ import annotations

import abc
import datetime
import logging
import os
import tempfile
import weakref
import zipfile
from contextlib import contextmanager
from functools import cached_property
from typing import TYPE_CHECKING, Iterable, Iterator

import xarray

from satimg.geometry import EdgeMode, Grid, Window, windows_at
from satimg.metadata import Field, Metadata, PatchMeta
from satimg.patch import Patch
from satimg.readers import CHUNK_PX
from satimg.tiling import read_window

if TYPE_CHECKING:  # pragma: no cover
    import PIL.Image

    from satimg.transform import Transformer

logger = logging.getLogger(__name__)


class Product(abc.ABC):
    def __init__(self, path: str):
        self._path = str(path)

    @property
    def path(self) -> str:
        return self._path

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._path!r})"

    # -- resources ------------------------------------------------
    def __enter__(self) -> "Product":
        return self

    def __exit__(self, *exc) -> None:
        self._close()

    def _close(self) -> None:
        """Close the rasters behind the cached :attr:`raw` / :attr:`visual` and
        drop them.

        Each view carries a ``_close`` that shuts every band it opened -- wired
        on in :func:`satimg.readers.merge_bands`, kept across the product's
        wrappers by :func:`satimg.readers.keep_open` -- so closing the two
        views is all the product needs to track. Does not touch a raw/visual
        view opened by an in-flight :meth:`patches` / :meth:`patches_at` call
        -- those close themselves when their own loop ends. Called on ``with``
        exit and by :func:`satimg.open_zip`; idempotent, and a later access
        rebuilds.
        """
        for name in ("raw", "visual"):
            da = self.__dict__.pop(name, None)
            if da is not None:
                try:
                    da.close()
                except Exception:  # pragma: no cover - best effort
                    logger.debug("error closing %s raster(s)", name, exc_info=True)

    # -- pixel views ------------------------------------------------
    @abc.abstractmethod
    def _open_raw(self, tile: int | tuple[int, int]) -> xarray.DataArray:
        """Open + merge every native band onto a common ``(band, y, x)`` grid,
        dask-chunked to ``tile`` (all bands in one chunk), native dtype, lazy.
        """

    @cached_property
    def raw(self) -> xarray.DataArray:
        """All native bands on a common ``(band, y, x)`` grid, native dtype,
        lazy, dask-chunked to a sane default (:data:`~satimg.readers.CHUNK_PX`).

        For a loop's own window size, iterate :meth:`patches` /
        :meth:`patches_at` instead: they open their own raw chunked to exactly
        the window size, and close it when the loop ends, rather than reusing
        (or resizing) this one.
        """
        return self._open_raw(CHUNK_PX)

    @cached_property
    def visual(self) -> xarray.DataArray:
        """The sensor's ``uint8`` visualisation, lazy: 3-band true colour for
        optical sensors, 1-band greyscale for SAR. Built from :attr:`raw`."""
        return self._render_visual(self.raw, CHUNK_PX)

    @abc.abstractmethod
    def _render_visual(
        self, raw: xarray.DataArray, tile: int | tuple[int, int]
    ) -> xarray.DataArray:
        """Build the visualisation array (sensor-specific) from ``raw``.

        ``tile`` is the same dask chunk size ``raw`` was opened with, for a
        subclass (e.g. Sentinel-2) whose visual reads an independent file
        rather than deriving from ``raw``, so it can be chunked to match."""

    # -- iteration ------------------------------------------------
    def _read_patch(
        self, raw: xarray.DataArray, visual: xarray.DataArray, window: Window
    ) -> Patch:
        """Slice ``raw`` / ``visual`` / :attr:`metadata` to ``window`` and
        bundle the result into a :class:`Patch`. The only place a ``Patch`` is
        ever built."""
        return Patch(
            window=window,
            raw=read_window(raw, window),
            visual=read_window(visual, window),
            meta=PatchMeta(self.metadata, window),
        )

    def patches(
        self,
        size: int | tuple[int, int] = 512,
        *,
        overlap: int | tuple[int, int] = 0,
        edge: EdgeMode = "pad",
    ) -> Iterator[Patch]:
        """Walk the product in windows of ``size``.

        Opens its own raw and visual view -- separate from :attr:`raw` /
        :attr:`visual` -- both dask-chunked to exactly ``size``, and closes
        them when the loop ends (return, break, or an exception), whichever
        comes first -- no explicit ``with``/``close()`` needed for this view.
        ``edge`` decides what happens to windows that run past the border (see
        :class:`~satimg.geometry.Grid`).
        """
        raw = self._open_raw(size)
        visual = self._render_visual(raw, size)
        try:
            grid = Grid(int(raw.sizes["x"]), int(raw.sizes["y"]), size, overlap, edge)
            for win in grid:
                yield self._read_patch(raw, visual, win)
        finally:
            raw.close()
            visual.close()

    def patches_at(
        self,
        points: Iterable[tuple[float, float]],
        size: int | tuple[int, int] = 512,
    ) -> Iterator[Patch]:
        """Walk the product at caller-supplied ``(col, row)`` points. See
        :func:`~satimg.geometry.windows_at`.

        Opens its own raw / visual dask-chunked to exactly ``size`` and closes
        them when the loop ends -- see :meth:`patches`.
        """
        raw = self._open_raw(size)
        visual = self._render_visual(raw, size)
        try:
            for win in windows_at(points, size):
                yield self._read_patch(raw, visual, win)
        finally:
            raw.close()
            visual.close()

    def __iter__(self) -> Iterator[Patch]:
        return iter(self.patches())

    # -- shape --------------------------------------------------
    @property
    def width(self) -> int:
        return int(self.raw.sizes["x"])

    @property
    def height(self) -> int:
        return int(self.raw.sizes["y"])

    @property
    def bands(self) -> int:
        return int(self.raw.sizes["band"])

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.bands, self.height, self.width

    @property
    def dtype(self):
        return self.raw.dtype

    # -- metadata ---------------------------------------------
    @property
    @abc.abstractmethod
    def transformer(self) -> "Transformer":
        """Pixel <-> lat/lon conversion for this product's grid."""

    @property
    @abc.abstractmethod
    def timestamp(self) -> datetime.datetime:
        """Acquisition time (scene centre)."""

    @property
    @abc.abstractmethod
    def footprint(self) -> dict:
        """GeoJSON-ish ``{"type": "Feature", "geometry": {...}}`` outline."""

    @abc.abstractmethod
    def thumbnail(self) -> "PIL.Image.Image":
        """The product's shipped quick-look image."""

    @cached_property
    def metadata(self) -> Metadata:
        """Per-pixel ancillary geometry (incidence / sun / view angles, ...).

        See :class:`~satimg.metadata.Metadata`. Empty for products that do not
        override :meth:`_read_metadata`.
        """
        return self._read_metadata()

    def _read_metadata(self) -> Metadata:
        """Build this product's :class:`~satimg.metadata.Metadata`. Override in a
        subclass; the default has no fields."""
        return Metadata({})

    def _field(self, rows, cols, values, *, name: str, units: str = "") -> Field:
        """Helper for :meth:`_read_metadata`: a :class:`~satimg.metadata.Field` on
        a coarse grid in this product's pixel coordinates.

        ``shape`` is passed as a callable, not ``(self.height, self.width)``
        directly -- those read :attr:`raw`, and evaluating them eagerly here
        would force every band open just to build a ``Field``, even if
        nothing ever reads its ``.grid`` / ``.corners()``. It closes over a
        ``weakref`` rather than ``self`` directly: :attr:`metadata` caches the
        ``Field``s it returns onto the product, so a plain closure would tie
        the product into a reference cycle through its own cached metadata.
        """
        product_ref = weakref.ref(self)

        def shape() -> tuple[int, int]:
            product = product_ref()
            assert product is not None, "product went away before its Field was read"
            return product.height, product.width

        return Field(
            rows,
            cols,
            values,
            name=name,
            units=units,
            shape=shape,
        )

    def bounds(self) -> tuple[float, float, float, float]:
        """``(min_lon, min_lat, max_lon, max_lat)`` of the footprint."""
        coords = self.footprint["geometry"]["coordinates"]
        while isinstance(coords[0][0], (list, tuple)):
            coords = coords[0]
        lons, lats = zip(*coords)
        return min(lons), min(lats), max(lons), max(lats)


def open(path: str) -> Product:
    """Open a product directory, returning the matching :class:`Product`."""
    from satimg.registry import resolve  # deferred: registry imports Product

    cls = resolve(path)
    logger.debug("opened %s as %s", path, cls.__name__)
    return cls(path)


@contextmanager
def open_zip(zip_path: str, dest: str | None = None) -> Iterator[Product]:
    """Extract a zipped product to a temp dir and yield it as a :class:`Product`.

    Use the ``with`` form -- the temp dir (under ``dest``, or the system
    default) and the product's open rasters are both released on exit. Read
    what you need inside the block; a lazy view cannot be rebuilt once the temp
    dir is gone.
    """
    with tempfile.TemporaryDirectory(dir=dest, ignore_cleanup_errors=True) as tmp:
        logger.debug("extracting %s to %s", zip_path, tmp)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(tmp)
        entries = [os.path.join(tmp, e) for e in os.listdir(tmp)]
        if not entries:
            raise ValueError(f"{zip_path} extracted to nothing")
        root = entries[0] if len(entries) == 1 and os.path.isdir(entries[0]) else tmp
        logger.debug("extracted product root: %s", root)
        with open(root) as product:
            yield product
