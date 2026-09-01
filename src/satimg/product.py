"""The :class:`Product` abstraction.

A product is a directory of satellite imagery + metadata. Whatever the sensor,
it offers the same three views on the pixels --

* ``product.raw``  -- every native band merged onto one grid, native dtype, lazy;
* ``product.rgb``  -- a 3-band ``uint8`` true-colour visualisation, lazy;
* ``product.gray`` -- a 1-band ``uint8`` visualisation, lazy;

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

Kind = str  # "raw" | "rgb" | "gray"


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
    def rgb(self) -> Raster:
        """3-band ``uint8`` visualisation, lazy."""
        return self._render_rgb()

    @cached_property
    def gray(self) -> Raster:
        """1-band ``uint8`` visualisation, lazy."""
        mono = self.rgb.array.mean("band", keep_attrs=True).round().clip(0, 255).astype("uint8")
        return Raster(mono, transform=self._transformer_or_none(), name="gray")

    @abc.abstractmethod
    def _render_rgb(self) -> Raster:
        """Build the true-colour raster (sensor-specific)."""

    def view(self, kind: Kind = "rgb") -> Raster:
        """Return one of the pixel views by name."""
        try:
            return {"raw": self.raw, "rgb": self.rgb, "gray": self.gray}[kind]
        except KeyError:
            raise ValueError(f"unknown view {kind!r}; expected raw/rgb/gray") from None

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

    def _transformer_or_none(self) -> "Transformer | None":
        try:
            return self.transformer
        except Exception:
            return None
