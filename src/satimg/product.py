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

Before a long loop -- especially over the whole image in patches -- call
``product.persist()`` to read ``raw`` and ``visual`` into memory once and drop
the open rasters; every window is then an in-memory slice. A product releases
its rasters when its ``with`` block exits, so use ``with satimg.open(path) as
product`` (and :func:`satimg.open_zip`) to loop over many without leaking
handles.
"""

from __future__ import annotations

import abc
import datetime
import logging
from functools import cached_property
from typing import TYPE_CHECKING, Iterable, Iterator

import xarray

from satimg import readers, tiling
from satimg.geometry import EdgeMode
from satimg.metadata import Field, Metadata
from satimg.tiling import Patch

if TYPE_CHECKING:  # pragma: no cover
    import PIL.Image

    from satimg.transform import Transformer

logger = logging.getLogger(__name__)


class Product(abc.ABC):
    def __init__(self, path: str):
        self._path = str(path)
        self._opened: list = []  # raster arrays backing raw / visual

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
        """Release every raster opened for this product and drop the cached
        :attr:`raw` / :attr:`visual` views.

        Called when the ``with`` block exits, and by :func:`satimg.open_zip`
        before its temp dir is removed. Idempotent; a later :attr:`raw` /
        :attr:`visual` access rebuilds, provided the files still exist.
        """
        while self._opened:
            try:
                self._opened.pop().close()
            except Exception:  # pragma: no cover - best effort
                logger.debug("error closing raster", exc_info=True)
        for name in ("raw", "visual"):
            self.__dict__.pop(name, None)

    def _open_band(self, path: str, *, chunk: int = 512) -> xarray.DataArray:
        """:func:`satimg.readers.open_band`, registering the opener for close."""
        da = readers.open_band(path, chunk=chunk)
        self._opened.append(da)
        return da

    def _merge_bands(
        self, paths, *, match: int | None = None, chunk: int = 512
    ) -> xarray.DataArray:
        """:func:`satimg.readers.merge_bands`, registering every opener for close."""
        merged, opened = readers.merge_bands(paths, match=match, chunk=chunk)
        self._opened.extend(opened)
        return merged

    def persist(self, *views: str) -> "Product":
        """Read the named pixel views into memory and release the source rasters.

        A following patch loop then reads in-memory slices instead of re-opening
        GDAL for every window -- call this before iterating, especially when
        tiling the whole image::

            product = satimg.open(path).persist()
            for patch in product.patches(512):
                ...

        With no arguments both :attr:`raw` and :attr:`visual` are persisted;
        pass ``"raw"`` / ``"visual"`` to pick. A view already in memory, or one
        pulled in as a dependency (``visual`` is built from ``raw`` for Landsat
        and Sentinel-1), is persisted too rather than left lazy over closed
        rasters. Returns ``self``.
        """
        for name in views or ("raw", "visual"):
            getattr(self, name)  # build it; may cache another view as a dependency
        loaded = {
            name: self.__dict__[name].load()
            for name in ("raw", "visual")
            if name in self.__dict__
        }
        self._close()
        self.__dict__.update(loaded)
        return self

    # -- pixel views ------------------------------------------------
    @property
    @abc.abstractmethod
    def raw(self) -> xarray.DataArray:
        """All native bands on a common ``(band, y, x)`` grid, native dtype, lazy."""

    @cached_property
    def visual(self) -> xarray.DataArray:
        """The sensor's ``uint8`` visualisation, lazy: 3-band true colour for
        optical sensors, 1-band greyscale for SAR."""
        return self._render_visual()

    @abc.abstractmethod
    def _render_visual(self) -> xarray.DataArray:
        """Build the visualisation array (sensor-specific)."""

    # -- iteration ------------------------------------------------
    def _align_view_chunks(self, size: int | tuple[int, int]) -> None:
        """Rechunk the built :attr:`raw` / :attr:`visual` views so each ``size``
        window is exactly one dask chunk -- the fast granularity for the patch
        loop about to run, and the reason a window read decodes one tile per
        band instead of the whole band.

        Runs once per ``patches`` / ``patches_at`` call, before iteration. A
        no-op for a view :meth:`persist` has already pulled into memory (those
        are plain slices, no chunking to align).
        """
        sw, sh = (size, size) if isinstance(size, int) else size
        for name in ("raw", "visual"):
            da = getattr(self, name)  # builds it lazily; no pixels read yet
            if getattr(da, "chunks", None) is not None:  # skip persisted (in-RAM) views
                self.__dict__[name] = da.chunk({"band": -1, "y": sh, "x": sw})

    def patches(
        self,
        size: int | tuple[int, int] = 512,
        *,
        overlap: int | tuple[int, int] = 0,
        edge: EdgeMode = "pad",
        batch: int | None = None,
    ) -> Iterator[Patch] | Iterator[list[Patch]]:
        """Walk :attr:`raw` in windows. See :func:`satimg.tiling.patches`.

        :attr:`raw` and :attr:`visual` are rechunked to ``size`` first, so every
        ``patch.raw`` / ``patch.visual`` / ``patch.values`` read decodes just the
        one tile it covers. Patches yielded here carry a lazy :attr:`Patch.meta`
        bound to this product's :attr:`metadata`.
        """
        self._align_view_chunks(size)
        return tiling.patches(
            self.raw, size, overlap=overlap, edge=edge, batch=batch,
            transformer=self.transformer, product=self,
        )

    def patches_at(
        self,
        points: Iterable[tuple[float, float]],
        size: int | tuple[int, int] = 512,
        *,
        batch: int | None = None,
    ) -> Iterator[Patch] | Iterator[list[Patch]]:
        """Walk :attr:`raw` at caller-supplied ``(col, row)`` points. See
        :func:`satimg.tiling.patches_at`.

        :attr:`raw` and :attr:`visual` are rechunked to ``size`` first (see
        :meth:`patches`). Patches yielded here carry a lazy :attr:`Patch.meta`
        bound to this product's :attr:`metadata`.
        """
        self._align_view_chunks(size)
        return tiling.patches_at(
            self.raw, points, size, batch=batch,
            transformer=self.transformer, product=self,
        )

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
            shape=(self.height, self.width),
        )

    def bounds(self) -> tuple[float, float, float, float]:
        """``(min_lon, min_lat, max_lon, max_lat)`` of the footprint."""
        coords = self.footprint["geometry"]["coordinates"]
        while isinstance(coords[0][0], (list, tuple)):
            coords = coords[0]
        lons, lats = zip(*coords)
        return min(lons), min(lats), max(lons), max(lats)
