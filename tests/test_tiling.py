import numpy
import pytest
import xarray

from satimg.geometry import Window
from satimg.tiling import as_band_yx, label_bands, read_window


def make_da(bands=3, height=200, width=300, dtype="uint8"):
    data = numpy.arange(bands * height * width, dtype=dtype).reshape(bands, height, width)
    return xarray.DataArray(data, dims=("band", "y", "x"), name="synthetic")


class TestAsBandYX:
    def test_promotes_2d_to_band_first(self):
        da = as_band_yx(xarray.DataArray(numpy.zeros((10, 20)), dims=("y", "x")))
        assert da.dims == ("band", "y", "x")
        assert da.shape == (1, 10, 20)

    def test_transposes_to_band_first(self):
        da = as_band_yx(xarray.DataArray(numpy.zeros((10, 20, 3)), dims=("y", "x", "band")))
        assert da.dims == ("band", "y", "x")

    def test_rejects_unnamed_dims(self):
        with pytest.raises(ValueError):
            as_band_yx(xarray.DataArray(numpy.zeros((3, 4, 5))))


class TestLabelBands:
    def test_assigns_band_coord(self):
        da = label_bands(make_da(bands=3), ["red", "green", "blue"])
        assert list(da.band.values) == ["red", "green", "blue"]
        assert da.sel(band="green").shape == (200, 300)
        # rides through a windowed read
        assert list(read_window(da, Window(0, 0, 8, 8)).band.values) == ["red", "green", "blue"]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            label_bands(make_da(bands=3), ["red", "green"])


class TestReadWindow:
    def test_interior_window(self):
        da = make_da()
        win = Window(10, 20, 50, 40)
        arr = read_window(da, win)
        assert arr.shape == (3, 40, 50)
        assert arr.attrs["col"] == 10 and arr.attrs["row"] == 20
        expected = da.isel(x=slice(10, 60), y=slice(20, 60)).to_numpy()
        numpy.testing.assert_array_equal(arr.to_numpy(), expected)

    def test_overhang_is_zero_padded(self):
        da = make_da(height=100, width=100)
        arr = read_window(da, Window(80, 90, 40, 40)).to_numpy()
        assert arr.shape == (3, 40, 40)
        assert (arr[:, 10:, :] == 0).all()  # rows past y=100
        assert (arr[:, :, 20:] == 0).all()  # cols past x=100

    def test_bare_yx_array(self):
        grid = xarray.DataArray(numpy.arange(50 * 50, dtype=float).reshape(50, 50), dims=("y", "x"))
        arr = read_window(grid, Window(40, 40, 20, 20))
        assert arr.dims == ("y", "x") and arr.shape == (20, 20)
        assert (arr.to_numpy()[10:, :] == 0).all()  # padded past y=50
        assert arr.attrs["col"] == 40 and arr.attrs["row"] == 40
