"""Behaviour shared by every concrete product, run against the minified test
data (pixels are zeroed; geometry, CRS and metadata are real)."""

import datetime
import itertools

import numpy
import pytest
import xarray

from satimg.geometry import Window
from satimg.products.landsat import BANDS as _LS_BANDS
from satimg.tiling import read_window

PRODUCTS = ["sentinel1_iw", "sentinel2_l1c", "landsat"]


@pytest.fixture(params=PRODUCTS)
def product(request):
    return request.getfixturevalue(request.param)


_S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
             "B08", "B8A", "B09", "B10", "B11", "B12"]


class TestViews:
    def test_raw_is_band_first_dataarray(self, product):
        raw = product.raw
        assert isinstance(raw, xarray.DataArray)
        assert raw.dims == ("band", "y", "x")
        assert product.shape == (product.bands, product.height, product.width)

    @pytest.mark.parametrize(
        "name,bands",
        [("sentinel1_iw", 1), ("sentinel2_l1c", 3), ("landsat", 3)],
    )
    def test_visual_is_uint8_with_sensor_band_count(self, request, name, bands):
        product = request.getfixturevalue(name)
        visual = product.visual
        assert visual.dtype == numpy.uint8
        assert visual.sizes["band"] == bands
        assert (visual.sizes["x"], visual.sizes["y"]) == (product.width, product.height)

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("sentinel1_iw", ["VV", "VH"]),
            ("sentinel2_l1c", _S2_BANDS),
            ("landsat", list(_LS_BANDS)),
        ],
    )
    def test_raw_band_coord_is_named(self, request, name, expected):
        raw = request.getfixturevalue(name).raw
        assert [str(b) for b in raw.band.values] == expected
        # named selection works and round-trips a coord
        assert raw.sel(band=expected[0]).shape == (raw.sizes["y"], raw.sizes["x"])

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("sentinel1_iw", ["amplitude"]),
            ("sentinel2_l1c", ["red", "green", "blue"]),
            ("landsat", ["red", "green", "blue"]),
        ],
    )
    def test_visual_band_coord_is_named(self, request, name, expected):
        assert [str(b) for b in request.getfixturevalue(name).visual.band.values] == expected

    def test_sentinel2_wavelength_coord(self, sentinel2_l1c):
        wl = sentinel2_l1c.raw.wavelength_nm
        assert wl.dims == ("band",) and wl.size == 13
        assert float(wl.sel(band="B08")) == pytest.approx(832.8, abs=0.1)
        assert float(wl.sel(band="B04")) == pytest.approx(664.6, abs=0.1)


class TestPatches:
    def test_iter_yields_patches_with_origin(self, product):
        first = next(iter(product))
        assert first.window.col == 0 and first.window.row == 0
        assert first.values.shape[0] == product.bands

    def test_patches_conserve_indices(self, product):
        seen = list(product.patches(512))
        assert seen[0].bounds == (0, 0, min(512, product.width), min(512, product.height))
        # every window lies inside the image and is placed on the stride grid
        for p in seen:
            assert 0 <= p.col < product.width
            assert 0 <= p.row < product.height
            assert p.col % 512 == 0 and p.row % 512 == 0

    def test_padded_patches_have_constant_shape(self, product):
        # bounded: full-res test scenes have thousands of tiles
        chan = product.bands
        head = itertools.islice(product.patches(256, edge="pad"), 8)
        assert {p.values.shape for p in head} == {(chan, 256, 256)}

    def test_overlap_batches(self, product):
        chan = product.bands
        groups = product.patches(256, overlap=32, edge="pad", batch=8)
        first = next(groups)
        assert len(first) == 8
        assert first[0].values.shape == (chan, 256, 256)
        assert first[1].window.col == 224  # stride = 256 - 32

    def test_raw_patch_matches_direct_read(self, product):
        win = Window(0, 0, 64, 64)
        patch = next(iter(product.patches(64)))
        numpy.testing.assert_array_equal(
            patch.values, read_window(product.raw, win).values
        )

    def test_patches_at_keeps_the_point_centred_including_at_a_padded_edge(self, product):
        # a point one pixel in from the top-left: raw and visual must still be
        # full-size (left/top zero-padded) with the point on the centre pixel.
        col, row = 1, 1
        p = next(product.patches_at([(col, row)], 128))
        assert p.raw.shape == (product.bands, 128, 128)
        assert p.visual.shape[1:] == (128, 128)
        assert p.window.col + 64 == col and p.window.row + 64 == row
        numpy.testing.assert_array_equal(
            p.raw.values, read_window(product.raw, p.window).values
        )
        numpy.testing.assert_array_equal(
            p.visual.values, read_window(product.visual, p.window).values
        )
        # the geo-centre of the patch is the requested pixel, not some rounded
        # window centre drifting off it
        want = product.transformer.rowcol_to_latlon((row, col))
        got = p.center_latlon
        assert got == pytest.approx((float(want[0, 0]), float(want[0, 1])), abs=1e-9)

    def test_patch_exposes_raw_visual_and_meta(self, product):
        p = next(product.patches_at([(600, 600)], 256))
        assert p.raw.dims == ("band", "y", "x")
        assert p.raw.shape == (product.bands, 256, 256)
        assert list(p.raw.band.values) == list(product.raw.band.values)
        assert p.visual.dtype == numpy.uint8
        assert p.visual.shape[1:] == (256, 256)
        assert p.visual.sizes["band"] == product.visual.sizes["band"]
        # window is shared: raw patch == direct read at the same window
        numpy.testing.assert_array_equal(
            p.raw.values, read_window(product.raw, p.window).values
        )
        assert set(p.meta.sample()) == set(product.metadata.fields)


class TestMetadata:
    def test_timestamp(self, product):
        assert isinstance(product.timestamp, datetime.datetime)

    def test_footprint_and_bounds(self, product):
        assert product.footprint["geometry"]["type"] == "Polygon"
        min_lon, min_lat, max_lon, max_lat = product.bounds()
        assert min_lon < max_lon and min_lat < max_lat

    def test_transformer_roundtrip(self, product):
        rowcol = numpy.array([[10.0, 10.0], [100.0, 250.0]])
        latlon = product.transformer.rowcol_to_latlon(rowcol)
        back = product.transformer.latlon_to_rowcol(latlon)
        numpy.testing.assert_allclose(back, rowcol, atol=3)

    def test_patch_center_latlon_in_footprint_bbox(self, product):
        min_lon, min_lat, max_lon, max_lat = product.bounds()
        lat, lon = next(iter(product)).center_latlon
        pad = 0.5
        assert min_lat - pad <= lat <= max_lat + pad
        assert min_lon - pad <= lon <= max_lon + pad


def test_sentinel1_mode(sentinel1_iw):
    assert sentinel1_iw.mode == "IW"


def test_thumbnail_opens(product):
    thumb = product.thumbnail()
    assert thumb.size[0] > 0 and thumb.size[1] > 0
