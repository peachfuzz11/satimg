import abc

from PIL.Image import Image

from common.image_slice import ImageSlice


class Viewable(abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def view_thumbnail(self, *args, **kwargs) -> Image:
        pass

    @abc.abstractmethod
    def view_full(self, *args, **kwargs) -> Image:
        pass

    @abc.abstractmethod
    def view_quicklook(self, *args, **kwargs) -> Image:
        pass

    @abc.abstractmethod
    def view_slice(self, image_slice: ImageSlice) -> Image:
        pass

    def _normalize_fn(self, x):
        return x
