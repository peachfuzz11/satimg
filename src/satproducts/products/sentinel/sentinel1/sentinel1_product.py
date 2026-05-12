import datetime
import os
import typing

import PIL.Image
import numpy
import rasterio
from PIL.Image import Image

from satproducts.common.image_slice import ImageSlice
from satproducts.products.base.product import Product
from satproducts.products.sentinel.sentinel1.service import sentinel1_service
from satproducts.transformers.base_transformer import BaseTransformer
from satproducts.transformers.gcp_transformer import GCPTransformer


class Sentinel1Product(Product):
    def __init__(self, product_path):
        super().__init__(product_path)
        manifest_data = sentinel1_service.get_manifest_data(self.product_path)
        self._product = sentinel1_service.open_dataarrays(product_path=product_path)
        self._footprint = manifest_data["footprint"]
        self._timestamp = manifest_data["timestamp"]
        self._width = self.product.sizes["x"]
        self._height = self.product.sizes["y"]
        self._channels = self.product.sizes["band"]
        with rasterio.open(self.product_path) as src:
            gcps, crs = src.gcps
            self._transformer = GCPTransformer(gcps=gcps, crs=crs)

    def tile(self, tile_size) -> typing.Generator[typing.Tuple[ImageSlice, numpy.ndarray], None, None]:
        xarr = self.product.chunk(dict(x=tile_size, y=tile_size, band=-1)).persist()
        for image_slice in self.slices(tile_size):
            m = xarr.isel(**image_slice.to_slice()).to_numpy().astype(numpy.float32)
            m = numpy.clip(m, 1e-12, None)
            m = 10 * numpy.log10(m)
            m = numpy.nan_to_num(m)
            m = numpy.mean(m, axis=0)
            m = 255 / (1 + numpy.exp(-((m - 20) * 0.18)))
            m = numpy.clip(m, 0, 255).astype("uint8")
            m = numpy.expand_dims(m, axis=0)
            m = numpy.repeat(m, 3, axis=0)
            yield image_slice, m

    def viewable(self, image_slice: ImageSlice = None, *args, **kwargs):
        arr = self.product
        if image_slice:
            arr = self.read_slice(image_slice)
        arr = 10 * numpy.log10(arr.where(arr > 0))
        arr = arr.fillna(0)
        arr = arr.mean(dim="band")
        arr = (255 * (1 / (1 + numpy.exp(-((arr - 20) * 0.18))))).clip(0, 255).astype("uint8")
        return arr

    def view(self, image_slice=None, **kwargs):
        return PIL.Image.fromarray(self.viewable(image_slice=image_slice).values)

    def view_thumbnail(self, *args, **kwargs) -> Image:
        p = os.path.join(self.product_path, "preview", "thumbnail.png")
        if not os.path.isfile(p):
            p = os.path.join(self.product_path, "preview", "quick-look.png")
        return PIL.Image.open(p)

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
    def transformer(self, *args, **kwargs) -> BaseTransformer:
        return self._transformer

    @property
    def timestamp(self) -> datetime.datetime:
        return self._timestamp

    @property
    def footprint(self) -> dict:
        return self._footprint
