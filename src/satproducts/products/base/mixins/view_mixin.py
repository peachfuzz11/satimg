import abc

from PIL.Image import Image

from satproducts.common.image_slice import ImageSlice


class ViewMixin(abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def view_thumbnail(self, *args, **kwargs) -> Image:
        pass

    @abc.abstractmethod
    def view(self, image_slice: ImageSlice = None) -> Image:
        pass
