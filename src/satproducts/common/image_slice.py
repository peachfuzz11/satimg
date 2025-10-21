import typing

import rasterio

from satproducts.common.slice import Slice
from satproducts.common.utils import slice_utils
from satproducts.common.utils.slice_utils import batched


class ImageSlice(Slice):

    def to_window(self) -> rasterio.windows.Window:
        return rasterio.windows.Window(self.i, self.j, self.width, self.height)

    def to_slice(self):
        return {'x': slice(self.i, self.i + self.width), 'y': slice(self.j, self.j + self.height)}

    def to_list(self):
        return [self.i, self.j, self.width, self.height]

    def __repr__(self):
        return f"{type(self)} i:{self.i} j:{self.j} width:{self.width} height:{self.height}"


def slice_generator(width, height, width_step, height_step=None) -> typing.Iterable["ImageSlice"]:
    for i, j, w, h in slice_utils.slices(
            width=width,
            height=height,
            width_step=width_step,
            height_step=height_step):
        yield ImageSlice(i=i, j=j, width=w, height=h)


def batched_slice_generator(batch_size, width, height, width_step, height_step=None) -> typing.Iterable[
    typing.List["ImageSlice"]]:
    for b in batched(slice_generator(width, height, width_step, height_step), batch_size):
        yield b


def patch_generator(patches, width_step, height_step=None) -> typing.Iterable["ImageSlice"]:
    for i, j, w, h in slice_utils.patches(patches=patches, width_step=width_step, height_step=height_step):
        yield ImageSlice(i=i, j=j, width=w, height=h)
