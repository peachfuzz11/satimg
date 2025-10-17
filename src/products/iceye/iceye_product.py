import os

import PIL
import numpy
import rasterio
from PIL.Image import Image

from common.image_slice import ImageSlice
from metadata.base_transformer import BaseTransformer
from products.base.image.raster_product import RasterProduct


class IceyeProduct(RasterProduct):

    def _extract_metadata(self):
        pass

    def validate(self):
        pass

    def view_thumbnail(self, *args, **kwargs) -> Image:
        image = self.view_quicklook().resize((200, 200))
        return image

    def view_full(self, *args, **kwargs) -> Image:
        with self:
            image = PIL.Image.fromarray((numpy.clip(self.read(1), 0, 5100.0) * 255.0 / 5100.0).astype('uint8'))
        return image

    def view_quicklook(self, *args, **kwargs) -> Image:
        image = PIL.Image.open(os.path.join(self.product_path,
                                            next(o for o in os.listdir(self.product_path) if
                                                 o.endswith('.png') and 'QUICKLOOK' in o)))
        return image

    def view_slice(self, image_slice: ImageSlice) -> Image:
        with self:
            image = self.read_slice(image_slice)[0]
            image = PIL.Image.fromarray((numpy.clip(image, 0, 5100.0) * 255.0 / 5100.0).astype('uint8'))
        return image

    def get_transformer(self, *args, **kwargs) -> BaseTransformer:
        return super().get_transformer(self.product_path)

    def open(self, *args, **kwargs):
        self._product = rasterio.open(
            os.path.join(self.product_path, next(
                o for o in os.listdir(self.product_path) if o.endswith('.tif') and o.startswith('ICEYE'))))
