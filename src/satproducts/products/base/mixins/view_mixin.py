import abc

import PIL.Image
import xarray
from PIL.Image import Image

from satproducts.common.image_slice import ImageSlice


class ViewMixin(abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def view_thumbnail(self, *args, **kwargs) -> Image:
        pass

    def view(self, image_slice: ImageSlice = None) -> Image:
        return PIL.Image.fromarray(self.viewable(image_slice=image_slice).values)

    @abc.abstractmethod
    def viewable(self, image_slice: ImageSlice = None) -> xarray.DataArray:
        pass
