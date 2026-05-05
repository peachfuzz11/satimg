import datetime
import os

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

    def viewable(self, image_slice: ImageSlice = None, *args, **kwargs):
        arr = self.product
        if image_slice:
            arr = self.read_slice(image_slice)
        arr = 10 * numpy.log10(arr.astype(float))
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
