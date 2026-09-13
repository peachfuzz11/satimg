"""The :class:`Product` abstraction.

A product is a directory of satellite imagery + metadata. Whatever the sensor,
it offers the same two views on the pixels --

* ``product.raw``    -- every native band merged onto one grid, native dtype, lazy;
* ``product.visual`` -- the sensor's ``uint8`` visualisation, lazy: a 3-band
  true-colour image for optical sensors, a 1-band greyscale image for SAR;

-- and one way to walk them::

    for patch in product.patches(512, overlap=64):
        tile = patch.values          # (band, y, x), origin at patch.col / patch.row
        lat, lon = patch.center_latlon

``for patch in product`` is shorthand for ``product.patches()``.

``patches`` / ``patches_at`` open their own raw / visual view, separate from
:attr:`Product.raw` / :attr:`Product.visual`, dask-chunked to exactly the
loop's window size -- so each patch read pulls a single tile -- and close it
when the loop ends (return, break, or an exception). ``product.raw`` /
``product.visual`` accessed directly (outside a loop) are cached on the
product and released when its ``with`` block exits, so use ``with
satimg.open(path) as product`` if you use those directly across many
products, to avoid leaking handles.

``satimg.open(path)`` (:meth:`Product.from_path`) takes either a product
directory or a zip archive -- detected by content, not by extension -- via a
:class:`~satimg.source.Source` factory (:func:`~satimg.source.open_source`).
A zip is read zip-native, straight off the archive with nothing ever
extracted: ``timestamp`` / ``footprint`` / ``transformer`` / ``thumbnail()``
/ ``metadata`` (all but Landsat's, which ships only as full rasters) work as
usual, but ``raw`` / ``visual`` / ``patches`` / ``patches_at`` raise
:class:`ZipNativeUnsupportedError` -- there is no pixel data to read without
extracting. For that, extract the archive first with :func:`open_zip`
(``extract=True``, the default), which returns a fully capable, directory-
backed product exactly like a plain ``satimg.open(directory)`` would.
"""

from __future__ import annotations

import abc
import datetime
import logging
import os
import tempfile
import zipfile
from contextlib import contextmanager
from functools import cached_property
from typing import TYPE_CHECKING, Iterable, Iterator

import xarray

from satimg import tiling
from satimg.geometry import EdgeMode
from satimg.metadata import Field, Metadata
from satimg.readers import CHUNK_PX
from satimg.source import Source
from satimg.tiling import Patch

if TYPE_CHECKING:  # pragma: no cover
    import PIL.Image

    from satimg.transform import Transformer

logger = logging.getLogger(__name__)


class ZipNativeUnsupportedError(NotImplementedError):
    """Raised by ``raw`` / ``visual`` / ``patches`` / ``patches_at`` -- and by
    Landsat's per-pixel angle ``metadata``, the one metadata piece with no
    XML/JSON equivalent -- on a zip-native product: one opened from a zip
    archive via :func:`open` (:meth:`Product.from_path`), or via
    :func:`open_zip`\\ ``(zip_path, extract=False)``. Those need real pixel
    data, which only exists once the archive is extracted --
    :func:`open_zip`\\ ``(zip_path, extract=True)`` (the default).
    """


class Product(abc.ABC):
    def __init__(self, source: Source):
        self._source = source
        self._path = source.path

    @classmethod
    def from_path(cls, path: str) -> "Product":
        """Open ``path``, returning the matching :class:`Product`.

        ``path`` may be a product directory or a zip archive -- resolved to a
        :class:`~satimg.source.Source` by the
        :func:`~satimg.source.open_source` factory (detected by content, not
        extension); the :class:`Product` subclass is then resolved by
        :func:`~satimg.registry.resolve` against the source's own identity
        name. A zip archive resolves to a zip-native source -- see the module
        docstring. Call on :class:`Product` itself (``Product.from_path``,
        or the ``satimg.open`` alias); the subclass this is called on is not
        consulted.
        """
        from satimg.registry import resolve  # deferred: registry imports Product
        from satimg.source import open_source

        source = open_source(path)
        resolved = resolve(source.name)
        logger.debug("opened %s as %s", path, resolved.__name__)
        return resolved(source)

    @property
    def path(self) -> str:
        return self._path

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._path!r})"

    def _require_extracted(self, what: str) -> None:
        """Guard for a raster-only code path: raise
        :class:`ZipNativeUnsupportedError` if this product is zip-native --
        opened from a zip archive (:func:`open` / :func:`Product.from_path`,
        or :func:`open_zip`\\ ``(..., extract=False)``) -- where no pixel
        data was ever written to disk."""
        if not self._source.is_directory:
            raise ZipNativeUnsupportedError(
                f"{what} needs the product extracted to disk -- extract the "
                f"archive first with satimg.open_zip(zip_path, extract=True) "
                f"(the default)"
            )

    # -- resources ------------------------------------------------
    def __enter__(self) -> "Product":
        return self

    def __exit__(self, *exc) -> None:
        self._close()

    def _close(self) -> None:
        """Close the rasters behind the cached :attr:`raw` / :attr:`visual`,
        drop them, and release :attr:`_source` (e.g. a zip-native source's
        open archive; a no-op for a plain directory).

        Each view carries a ``_close`` that shuts every band it opened -- wired
        on in :func:`satimg.readers.merge_bands`, kept across the product's
        wrappers by :func:`satimg.readers.keep_open` -- so closing the two
        views is all the product needs to track, on top of the source itself.
        Does not touch a raw/visual view opened by an in-flight
        :meth:`patches` / :meth:`patches_at` call -- those close themselves
        when their own loop ends. Called on ``with`` exit; idempotent (both
        this and :meth:`~satimg.source.Source.close` are). A later access
        rebuilds :attr:`raw` / :attr:`visual` for a directory-backed product,
        but a zip-native one has nothing left to rebuild from.
        """
        for name in ("raw", "visual"):
            da = self.__dict__.pop(name, None)
            if da is not None:
                try:
                    da.close()
                except Exception:  # pragma: no cover - best effort
                    logger.debug("error closing %s raster(s)", name, exc_info=True)
        self._source.close()

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
    def patches(
        self,
        size: int | tuple[int, int] = 512,
        *,
        overlap: int | tuple[int, int] = 0,
        edge: EdgeMode = "pad",
        batch: int | None = None,
    ) -> Iterator[Patch] | Iterator[list[Patch]]:
        """Walk the product in windows of ``size``. See
        :func:`satimg.tiling.patches`.

        Opens its own raw and visual view -- separate from :attr:`raw` /
        :attr:`visual` -- both dask-chunked to exactly ``size``, so every
        ``patch.raw`` / ``patch.visual`` / ``patch.values`` read decodes just
        the one tile it covers. Both close when the loop ends (return, break,
        or an exception), whichever comes first -- no explicit
        ``with``/``close()`` needed for this view. Patches yielded here carry
        a lazy :attr:`Patch.meta` bound to this product's :attr:`metadata`.
        """
        raw = self._open_raw(size)
        visual = self._render_visual(raw, size)
        try:
            yield from tiling.patches(
                raw, size, overlap=overlap, edge=edge, batch=batch,
                transformer=self.transformer, product=self, visual=visual,
            )
        finally:
            raw.close()
            visual.close()

    def patches_at(
        self,
        points: Iterable[tuple[float, float]],
        size: int | tuple[int, int] = 512,
        *,
        batch: int | None = None,
    ) -> Iterator[Patch] | Iterator[list[Patch]]:
        """Walk the product at caller-supplied ``(col, row)`` points. See
        :func:`satimg.tiling.patches_at`.

        Opens its own raw / visual dask-chunked to exactly ``size`` and closes
        them when the loop ends -- see :meth:`patches`. Patches yielded here
        carry a lazy :attr:`Patch.meta` bound to this product's
        :attr:`metadata`.
        """
        raw = self._open_raw(size)
        visual = self._render_visual(raw, size)
        try:
            yield from tiling.patches_at(
                raw, points, size, batch=batch,
                transformer=self.transformer, product=self, visual=visual,
            )
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
        a coarse grid in this product's pixel coordinates."""
        return Field(
            rows,
            cols,
            values,
            name=name,
            units=units,
            transform=self.transformer,
            shape=self._metadata_grid_shape(),
        )

    def _metadata_grid_shape(self) -> tuple[int, int]:
        """``(height, width)`` of the full product grid, backing a
        :class:`~satimg.metadata.Field`'s ``.grid`` / ``.corners()``.

        Default: the real raster shape, i.e. ``self.height, self.width`` --
        which forces ``raw`` open, so raises :class:`ZipNativeUnsupportedError`
        in zip-native mode. Override with a value already known from parsed
        metadata (no raster access) to keep this working zip-native too --
        Sentinel-1/-2 do, from their manifest/annotation XML; Landsat's
        per-pixel metadata is itself extraction-only (see
        :meth:`_require_extracted`), so it never needs to.
        """
        return self.height, self.width

    def bounds(self) -> tuple[float, float, float, float]:
        """``(min_lon, min_lat, max_lon, max_lat)`` of the footprint."""
        coords = self.footprint["geometry"]["coordinates"]
        while isinstance(coords[0][0], (list, tuple)):
            coords = coords[0]
        lons, lats = zip(*coords)
        return min(lons), min(lats), max(lons), max(lats)


def open(path: str) -> Product:
    """Open a product, returning the matching :class:`Product`. See
    :meth:`Product.from_path`."""
    return Product.from_path(path)


@contextmanager
def open_zip(
    zip_path: str, dest: str | None = None, extract: bool = True
) -> Iterator[Product]:
    """Open a zipped product. Use the ``with`` form either way.

    With ``extract=True`` (the default), the archive is fully extracted to a
    temp dir (under ``dest``, or the system default) and the returned
    :class:`Product` supports everything a directory-backed one does,
    including ``raw`` / ``visual`` / ``patches`` / ``patches_at``. Both the
    temp dir and the product's open rasters are released on exit; read what
    you need inside the block -- a lazy view cannot be rebuilt once the temp
    dir is gone.

    With ``extract=False``, nothing is ever written to disk (``dest`` is
    unused) -- same as plain ``satimg.open(zip_path)``
    (:meth:`Product.from_path`), which resolves a zip archive to this same
    zip-native mode. The returned product's ``timestamp`` / ``footprint`` /
    ``transformer`` / ``thumbnail()`` (and, except for Landsat, ``metadata``)
    work straight off the zip; ``raw`` / ``visual`` / ``patches`` /
    ``patches_at`` (and Landsat's ``metadata``) raise
    :class:`ZipNativeUnsupportedError` -- use ``extract=True`` for those.
    """
    if not extract:
        from satimg.registry import resolve  # deferred: registry imports Product
        from satimg.source import zip_source

        source = zip_source(zip_path)
        cls = resolve(source.name)
        with cls(source) as product:
            yield product
        return

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
