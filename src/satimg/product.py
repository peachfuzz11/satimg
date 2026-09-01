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
from typing import TYPE_CHECKING, Iterator

from satimg.geometry import EdgeMode
from satimg.raster import Patch, Raster

if TYPE_CHECKING:  # pragma: no cover
    import PIL.Image

    from satimg.transform import Transformer

Kind = str  # "raw" | "visual"


class Product(abc.ABC):
    #: window size used by bare ``iter(product)`` / ``product.patches()``.
    patch_size: int = 512

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
    def raw(self) -> Raster:
        """All native bands on a common grid, native dtype, lazy."""

    @cached_property
    def visual(self) -> Raster:
        """The sensor's ``uint8`` visualisation, lazy: 3-band true colour for
        optical sensors, 1-band greyscale for SAR."""
        return self._render_visual()

    @abc.abstractmethod
    def _render_visual(self) -> Raster:
        """Build the visualisation raster (sensor-specific)."""

    def view(self, kind: Kind = "visual") -> Raster:
        """Return one of the pixel views by name."""
        try:
            return {"raw": self.raw, "visual": self.visual}[kind]
        except KeyError:
            raise ValueError(f"unknown view {kind!r}; expected raw/visual") from None

    # -- iteration ------------------------------------------------
    def patches(
        self,
        size: int | tuple[int, int] | None = None,
        *,
        overlap: int | tuple[int, int] = 0,
        edge: EdgeMode = "trim",
        kind: Kind = "raw",
        batch: int | None = None,
    ) -> Iterator[Patch] | Iterator[list[Patch]]:
        """Walk a pixel view in windows. See :meth:`Raster.patches`."""
        return self.view(kind).patches(
            size or self.patch_size, overlap=overlap, edge=edge, batch=batch
        )

    def __iter__(self) -> Iterator[Patch]:
        return self.patches()

    # -- shape --------------------------------------------------
    @property
    def width(self) -> int:
        return self.raw.width

    @property
    def height(self) -> int:
        return self.raw.height

    @property
    def bands(self) -> int:
        return self.raw.bands

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

    def bounds(self) -> tuple[float, float, float, float]:
        """``(min_lon, min_lat, max_lon, max_lat)`` of the footprint."""
        coords = self.footprint["geometry"]["coordinates"]
        while isinstance(coords[0][0], (list, tuple)):
            coords = coords[0]
        lons, lats = zip(*coords)
        return min(lons), min(lats), max(lons), max(lats)
