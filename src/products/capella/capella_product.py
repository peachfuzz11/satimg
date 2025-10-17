import abc
import typing

from PIL.Image import Image

from common.image_slice import ImageSlice
from products.base.image.raster_product import RasterProduct


class CapellaProduct(RasterProduct, abc.ABC):
    def __init__(self, product_path):
        super().__init__(product_path)

    def _init(self):
        with self:
            self._width = self.product.sizes["x"]
            self._height = self.product.sizes["y"]
            self._nr_bands = self.product.sizes["band"]

            self._extract_metadata()

            self.validate()

    def view_thumbnail(self, *args, **kwargs) -> Image:
        raise NotImplementedError("Thumbnail view is not implemented for Capella products.")

    def view_full(self, band: typing.Union[int, str] = 0, stride: int = 1, *args, **kwargs) -> Image:
        pass

    def view_quicklook(self, *args, **kwargs) -> Image:
        raise NotImplementedError("Quicklook view is not implemented for Capella products.")

    def view_slice(self, image_slice: ImageSlice, band: typing.Union[int, str] = 0) -> Image:
        pass

    def _extract_metadata(self):
        pass

    @abc.abstractmethod
    def validate(self):
        pass
