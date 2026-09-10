import numpy
import pytest
import xarray

from satimg.geometry import Window
from satimg.tiling import (
    Patch,
    as_band_yx,
    label_bands,
    patches,
    patches_at,
    read_window,
)


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


class TestPatches:
    def test_tile_the_image_and_keep_origin(self):
        da = make_da(height=200, width=300)
        got = list(patches(da, 128, edge="pad"))
        assert len(got) == 2 * 3
        for p in got:
            assert isinstance(p, Patch)
            assert p.values.shape == (3, 128, 128)
            if p.col < 300 and p.row < 200:
                assert p.values[0, 0, 0] == da.to_numpy()[0, p.row, p.col]

    def test_reassembly_is_lossless_with_trim(self):
        da = make_da(bands=1, height=150, width=170)
        canvas = numpy.zeros((150, 170), dtype=da.dtype)
        for p in patches(da, 64, edge="trim"):
            ys, xs = p.window.slices()
            canvas[ys, xs] = p.values[0]
        numpy.testing.assert_array_equal(canvas, da.to_numpy()[0])

    def test_batched(self):
        groups = list(patches(make_da(), 128, edge="pad", batch=4))
        assert [len(g) for g in groups] == [4, 2]

    def test_default_edge_is_pad(self):
        da = make_da(height=200, width=300)
        assert {p.values.shape for p in patches(da, 128)} == {(3, 128, 128)}

    def test_over_a_derived_array(self):
        da = make_da(dtype="float32")
        got = list(patches(da * 2, 128))
        numpy.testing.assert_array_equal(got[0].values, (da * 2).to_numpy()[:, :128, :128])


class TestPatchesAt:
    def test_centers_on_points(self):
        da = make_da(height=200, width=300)
        points = [(150, 100), (40, 30)]
        got = list(patches_at(da, points, 64))
        assert len(got) == 2
        for p, (col, row) in zip(got, points):
            assert isinstance(p, Patch)
            assert p.values.shape == (3, 64, 64)
            assert p.window.center == (col, row)
            numpy.testing.assert_array_equal(
                p.values[:, 32, 32], da.to_numpy()[:, row, col]
            )

    def test_out_of_bounds_is_padded(self):
        (p,) = list(patches_at(make_da(height=100, width=100), [(2, 2)], 32))
        arr = p.values
        assert arr.shape == (3, 32, 32)
        assert (arr[:, :14, :] == 0).all()  # rows above y=0
        assert (arr[:, :, :14] == 0).all()  # cols left of x=0

    @pytest.mark.parametrize(
        "col,row,size",
        [
            (150, 100, 64),   # interior
            (3, 4, 64),       # overhangs the origin -- left/top zero-pad
            (297, 198, 64),   # overhangs the far edge -- right/bottom zero-pad
            (150, 100, 61),   # odd size, still dead-centre
            (30, 30, 500),    # window larger than the image -- padded all round
            (150.4, 99.6, 64),  # fractional point rounds to the centre pixel
        ],
    )
    def test_point_stays_dead_centre_through_the_zero_pad(self, col, row, size):
        da = make_da(bands=2, height=200, width=300, dtype="int32")
        (p,) = list(patches_at(da, [(col, row)], size))
        arr = p.values
        assert arr.shape == (2, size, size)
        c, r = int(round(col)), int(round(row))
        numpy.testing.assert_array_equal(
            arr[:, size // 2, size // 2], da.to_numpy()[:, r, c]
        )
        # padded-out region reads as zero, real data does not (arange starts at 0
        # on band 0, so check band 1 which is strictly positive)
        assert arr[1, size // 2, size // 2] != 0

    def test_batched(self):
        points = [(x, x) for x in range(10, 100, 10)]  # 9 points
        groups = list(patches_at(make_da(), points, 32, batch=4))
        assert [len(g) for g in groups] == [4, 4, 1]


class TestPatch:
    def test_without_transformer_center_latlon_raises(self):
        p = Patch(make_da(), Window(0, 0, 4, 4))
        with pytest.raises(AttributeError):
            p.center_latlon

    def test_without_product_meta_raises(self):
        p = Patch(make_da(), Window(0, 0, 4, 4))
        with pytest.raises(AttributeError):
            p.meta

    def test_without_product_raw_visual_raise(self):
        p = next(patches(make_da(), 4))
        with pytest.raises(AttributeError):
            p.raw
        with pytest.raises(AttributeError):
            p.visual

    def test_image_from_bare_yx(self):
        grid = xarray.DataArray(
            numpy.zeros((8, 8), dtype="uint8"), dims=("y", "x")
        )
        (p,) = list(patches_at(grid, [(4, 4)], 4))
        img = p.image()
        assert img.size == (4, 4)

    def test_image_from_band_yx(self):
        (p,) = list(patches_at(make_da(bands=3), [(50, 50)], 8))
        assert p.image().size == (8, 8)
