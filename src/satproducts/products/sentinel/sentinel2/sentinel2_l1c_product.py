import datetime
import os
import typing

import PIL.Image
import numpy
import rasterio
import xarray
from PIL.Image import Image

from satproducts.common.image_slice import ImageSlice
from satproducts.products.base.product import Product
from satproducts.products.sentinel.sentinel2.service import sentinel2_service
from satproducts.transformers.base_transformer import BaseTransformer
from satproducts.transformers.transformer import Transformer


class Sentinel2L1CProduct(Product):

    def __init__(self, product_path):
        super().__init__(product_path=product_path)
        mtd_msil1c_data = sentinel2_service.get_mtd_msil1c_data(product_path)
        bands = mtd_msil1c_data["bands"]
        self._footprint = mtd_msil1c_data["footprint"]
        mtd_tl_data = sentinel2_service.get_mtd_tl_data(product_path)
        self._timestamp = mtd_tl_data["timestamp"]

        das = sentinel2_service.open_dataarrays(bands)
        self._product = sentinel2_service.reindex_and_concatenate_dataarrays(das)
        tci = [b["file_path"] for b in bands if b["physical_band"] is None][0]
        self._tci = xarray.open_dataarray(tci).astype("uint8")

        self._width = self.product.sizes['x']
        self._height = self.product.sizes['y']
        self._channels = self.product.sizes['band']
        with rasterio.open(bands[1]['file_path']) as src:
            self._transformer = Transformer(transform=src.transform, crs=src.crs)

    def tile(self, tile_size) -> typing.Generator[typing.Tuple[ImageSlice, numpy.ndarray], None, None]:
        tci = self._tci.chunk(dict(x=tile_size, y=tile_size, band=-1))
        for image_slice in self.slices(tile_size):
            yield image_slice, tci[image_slice.to_slice()].to_numpy().astype("uint8")

    def viewable(self, image_slice: ImageSlice = None):
        tci = self._tci
        if image_slice:
            tci = tci[image_slice.to_slice()]
        tci = tci.transpose("y", "x", "band")
        return tci

    def view_thumbnail(self, *args, **kwargs) -> Image:
        image = PIL.Image.open(
            os.path.join(self.product_path, next(o for o in os.listdir(self.product_path) if o.endswith("-ql.jpg"))))
        return image

    @property
    def transformer(self, *args, **kwargs) -> BaseTransformer:
        return self._transformer

    @property
    def product(self):
        return self._product

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def channels(self):
        return self._channels

    @property
    def timestamp(self) -> datetime.datetime:
        return self._timestamp

    @property
    def footprint(self) -> dict:
        return self._footprint
