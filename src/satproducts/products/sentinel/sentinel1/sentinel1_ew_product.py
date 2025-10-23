from PIL.Image import Image

from satproducts.common.image_slice import ImageSlice
from satproducts.products.sentinel.sentinel1.sentinel1_product import Sentinel1Product


class Sentinel1EWProduct(Sentinel1Product):

    def viewable(self, image_slice: ImageSlice = None, a_min=0, a_max=5010., *args, **kwargs) -> Image:
        return super().view(image_slice=image_slice, a_min=a_min, a_max=a_max, *args, **kwargs)
