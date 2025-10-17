import abc
import typing

from common.image_slice import ImageSlice, slice_generator, patch_generator


class Sliceable(abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def read_slice(self, image_slice: ImageSlice):
        pass

    @property
    @abc.abstractmethod
    def height(self):
        pass

    @property
    @abc.abstractmethod
    def width(self):
        pass

    def slices(self, width_step: int, height_step: int = None) -> typing.Iterable[ImageSlice]:
        height_step = height_step or width_step
        return slice_generator(self.width, self.height, width_step, height_step)

    def patches(self, patches: typing.Generator[typing.Tuple[int, int], None, None], width_step: int,
                height_step: int = None) -> typing.Iterable[ImageSlice]:
        height_step = height_step or width_step
        return patch_generator(patches, width_step, height_step)
