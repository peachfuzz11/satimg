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
"""

from __future__ import annotations

import abc
import datetime
from functools import cached_property
from typing import TYPE_CHECKING, Iterable, Iterator

import xarray

from satimg import tiling
from satimg.geometry import EdgeMode
from satimg.metadata import Field, Metadata
from satimg.tiling import Patch

if TYPE_CHECKING:  # pragma: no cover
    import PIL.Image

    from satimg.transform import Transformer


class Product(abc.ABC):
    def __init__(self, path: str):
        self._path = str(path)

    @property
    def path(self) -> str:
        return self._path

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._path!r})"

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
    def patches(
        self,
        size: int | tuple[int, int] = 512,
        *,
        overlap: int | tuple[int, int] = 0,
        edge: EdgeMode = "pad",
        batch: int | None = None,
    ) -> Iterator[Patch] | Iterator[list[Patch]]:
        """Walk :attr:`raw` in windows. See :func:`satimg.tiling.patches`.

        Patches yielded here carry a lazy :attr:`Patch.meta` bound to this
        product's :attr:`metadata`. To walk another array (``visual``, a derived
        one) use ``satimg.patches(da, ..., transformer=product.transformer)``.
        """
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

        Patches yielded here carry a lazy :attr:`Patch.meta` bound to this
        product's :attr:`metadata`.
        """
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
