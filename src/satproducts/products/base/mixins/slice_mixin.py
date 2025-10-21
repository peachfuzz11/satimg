import abc
import typing

from satproducts.common.image_slice import ImageSlice, slice_generator, patch_generator


class SliceMixin(abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def read_slice(self, image_slice: ImageSlice):
        subset = self.product.isel(**image_slice.to_slice())
        subset.attrs["i"] = image_slice.i
        subset.attrs["j"] = image_slice.j
        return subset

    @property
    @abc.abstractmethod
    def product(self):
        pass

    @property
    @abc.abstractmethod
    def height(self):
        pass

    @property
    @abc.abstractmethod
    def width(self):
        pass

    @property
    @abc.abstractmethod
    def channels(self):
        pass

    def slices(self, width_step: int, height_step: int = None) -> typing.Iterable[ImageSlice]:
        height_step = height_step or width_step
        return slice_generator(self.width, self.height, width_step, height_step)

    def patches(self, patches: typing.Generator[typing.Tuple[int, int], None, None], width_step: int,
                height_step: int = None) -> typing.Iterable[ImageSlice]:
        height_step = height_step or width_step
        return patch_generator(patches, width_step, height_step)
