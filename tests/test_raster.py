import numpy
import pytest
import xarray

from satimg.geometry import Window
from satimg.raster import Patch, Raster


def make_raster(bands=3, height=200, width=300, dtype="uint8"):
    data = numpy.arange(bands * height * width, dtype=dtype).reshape(bands, height, width)
    da = xarray.DataArray(data, dims=("band", "y", "x"))
    return Raster(da, name="synthetic")


class TestRaster:
    def test_shape(self):
        r = make_raster()
        assert (r.bands, r.height, r.width) == (3, 200, 300)
        assert r.shape == (3, 200, 300)

    def test_promotes_2d_to_band_first(self):
        da = xarray.DataArray(numpy.zeros((10, 20)), dims=("y", "x"))
        r = Raster(da)
        assert r.shape == (1, 10, 20)
        assert r.array.dims == ("band", "y", "x")

    def test_rejects_unnamed_dims(self):
        with pytest.raises(ValueError):
            Raster(xarray.DataArray(numpy.zeros((3, 4, 5))))

    def test_read_interior_window(self):
        r = make_raster()
        win = Window(10, 20, 50, 40)
        arr = r.read(win)
        assert arr.shape == (3, 40, 50)
        assert arr.attrs["col"] == 10 and arr.attrs["row"] == 20
        expected = r.array.isel(x=slice(10, 60), y=slice(20, 60)).to_numpy()
        numpy.testing.assert_array_equal(arr.to_numpy(), expected)

    def test_read_overhang_is_zero_padded(self):
        r = make_raster(height=100, width=100)
        arr = r.read(Window(80, 90, 40, 40)).to_numpy()
        assert arr.shape == (3, 40, 40)
        assert (arr[:, 10:, :] == 0).all()  # rows past y=100
        assert (arr[:, :, 20:] == 0).all()  # cols past x=100

    def test_patches_tile_the_image_and_keep_origin(self):
        r = make_raster(height=200, width=300)
        patches = list(r.patches(128, edge="pad"))
        assert len(patches) == 2 * 3
        for p in patches:
            assert isinstance(p, Patch)
            assert p.values.shape == (3, 128, 128)
            block = p.values
            top_left = block[0, 0, 0]
            if p.col < 300 and p.row < 200:
                assert top_left == r.array.to_numpy()[0, p.row, p.col]

    def test_patch_reassembly_is_lossless_with_trim(self):
        r = make_raster(bands=1, height=150, width=170)
        canvas = numpy.zeros((150, 170), dtype=r.dtype)
        for p in r.patches(64, edge="trim"):
            ys, xs = p.window.slices()
            canvas[ys, xs] = p.values[0]
        numpy.testing.assert_array_equal(canvas, r.array.to_numpy()[0])

    def test_batched_patches(self):
        r = make_raster()
        groups = list(r.patches(128, edge="pad", batch=4))
        assert [len(g) for g in groups] == [4, 2]

    def test_iter_uses_defaults(self):
        r = make_raster(height=600, width=600)
        assert sum(1 for _ in r) == len(list(r.patches(512)))

    def test_map_is_lazy_chain(self):
        r = make_raster(dtype="float32")
        doubled = r.map(lambda d: d * 2, name="x2")
        assert doubled.name == "x2"
        numpy.testing.assert_array_equal(doubled.array.to_numpy(), r.array.to_numpy() * 2)

    def test_hwc(self):
        assert make_raster(bands=3).hwc().dims == ("y", "x", "band")
        assert make_raster(bands=1).hwc().dims == ("y", "x")
