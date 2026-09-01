"""Behaviour shared by every concrete product, run against the minified test
data (pixels are zeroed; geometry, CRS and metadata are real)."""

import datetime
import itertools

import numpy
import pytest

from satimg.geometry import Window
from satimg.raster import Raster

PRODUCTS = ["sentinel1_iw", "sentinel2_l1c", "landsat"]


@pytest.fixture(params=PRODUCTS)
def product(request):
    return request.getfixturevalue(request.param)


class TestViews:
    def test_raw_is_band_first_raster(self, product):
        raw = product.raw
        assert isinstance(raw, Raster)
        assert raw.array.dims == ("band", "y", "x")
        assert raw.shape == (product.bands, product.height, product.width)

    def test_rgb_is_uint8_three_band(self, product):
        rgb = product.rgb
        assert rgb.dtype == numpy.uint8
        assert rgb.bands == 3
        assert (rgb.width, rgb.height) == (product.width, product.height)

    def test_gray_is_uint8_single_band(self, product):
        assert product.gray.dtype == numpy.uint8
        assert product.gray.bands == 1

    def test_view_lookup(self, product):
        assert product.view("raw") is product.raw
        with pytest.raises(ValueError):
            product.view("nope")


class TestPatches:
    def test_iter_yields_patches_with_origin(self, product):
        first = next(iter(product))
        assert first.window.col == 0 and first.window.row == 0
        assert first.values.shape[0] == product.bands

    def test_patches_conserve_indices(self, product):
        seen = list(product.patches(512, kind="rgb"))
        assert seen[0].bounds == (0, 0, min(512, product.width), min(512, product.height))
        # every window lies inside the image and is placed on the stride grid
        for p in seen:
            assert 0 <= p.col < product.width
            assert 0 <= p.row < product.height
            assert p.col % 512 == 0 and p.row % 512 == 0

    def test_padded_patches_have_constant_shape(self, product):
        # bounded: full-res test scenes have thousands of tiles
        head = itertools.islice(product.patches(256, edge="pad", kind="rgb"), 8)
        assert {p.values.shape for p in head} == {(3, 256, 256)}

    def test_overlap_batches(self, product):
        groups = product.patches(256, overlap=32, edge="pad", batch=8, kind="gray")
        first = next(groups)
        assert len(first) == 8
        assert first[0].values.shape == (1, 256, 256)
        assert first[1].window.col == 224  # stride = 256 - 32

    def test_raw_patch_matches_direct_read(self, product):
        win = Window(0, 0, 64, 64)
        patch = next(iter(product.patches(64, kind="raw")))
        numpy.testing.assert_array_equal(patch.values, product.raw.values(win))


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
