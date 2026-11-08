"""Sentinel-2 Level-1C reader."""

from __future__ import annotations

import datetime

from satimg.metadata.collection import Metadata
from satimg.product import Product
from satimg.products.sentinel2 import files
from satimg.readers.raster import GDALRaster
from satimg.readers.stack import BandStack
from satimg.readers.thumbnail import load_thumbnail
from satimg.registry import register
from satimg.source.base import Source
from satimg.transform.affine import AffineTransformer


@register(r"^S2[ABCD]_MSIL1C_\d{8}T\d{6}_.*_\d{8}T\d{6}\.SAFE$")
class Sentinel2L1CProduct(Product):
    """``raw`` is the 13 reflectance bands resampled to the 10 m grid; ``visual``
    is the shipped 3-band True Colour Image. ``metadata`` carries the ``MTD_TL.xml``
    angle grids (``sun_zenith``, ``sun_azimuth``, ``view_zenith``, ``view_azimuth``);
    the viewing grids are the per-detector grids merged. SAFE parsing itself
    lives in :mod:`satimg.products.sentinel2.files`."""

    def __init__(self, source: Source):
        super().__init__(source)
        self._meta = files.parse_mtd(self._source)
        self._tl = files.parse_tl(self._source)
        self._timestamp = self._tl["timestamp"]
        self._gsd = files.BAND_GSD_M[self._meta["band_names"][1]]
        self._transformer = AffineTransformer(self._tl["transforms"][self._gsd], self._tl["crs"])

    def _raw_bands(self) -> BandStack:
        # every band resampled onto band_names[1]'s 10 m grid
        return BandStack(
            [GDALRaster(p) for p in self._meta["reflectance"]], self._meta["band_names"], self._meta["wavelengths"], match=1
        )

    def _visual_bands(self) -> BandStack:
        # the true-colour image is a file of its own: no raw needed
        return BandStack([GDALRaster(self._meta["tci"], dtype="uint8")], ("red", "green", "blue"))

    def _read_metadata(self) -> Metadata:
        fields = {
            name: self._field(*self._tl["axes"][name], grid, name=name, units="degrees")
            for name, grid in self._tl["grids"].items()
        }
        return Metadata(fields, self._tl["attrs"])

    @property
    def height(self) -> int:
        """From ``MTD_TL.xml``'s ``Tile_Geocoding/Size`` at ``band_names[1]``'s
        resolution (matches ``raw`` exactly) rather than the base
        ``int(self.raw.sizes["y"])`` -- so ``metadata``'s fields get a real
        ``.grid`` / ``.corners()`` even zip-native, with no ``raw`` open
        needed."""
        return self._tl["pixel_shapes"][self._gsd][0]

    @property
    def width(self) -> int:
        """See :attr:`height`."""
        return self._tl["pixel_shapes"][self._gsd][1]

    @property
    def transformer(self) -> AffineTransformer:
        return self._transformer

    @property
    def timestamp(self) -> datetime.datetime:
        return self._timestamp

    @property
    def footprint(self) -> dict:
        return self._meta["footprint"]

    def thumbnail(self):
        name = next(o for o in self._source.listdir() if o.endswith("-ql.jpg"))
        return load_thumbnail(self._source, name)
