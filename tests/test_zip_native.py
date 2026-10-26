"""``satimg.open(zip_path)`` (a zip archive, detected by content) reads a
product straight off the archive, with nothing extracted to disk -- metadata
and pixels alike, compared here against the same product opened normally
(directory-backed), not golden values. Archives are built stored, like the
providers' downloads; one deflated archive checks that works too."""

import os
import zipfile

import numpy
import pytest

import satimg
from tests.conftest import PRODUCT_PATHS
from tests.test_resources import _build_zip

PRODUCTS = ["sentinel1_iw", "sentinel2_l1c", "landsat"]


@pytest.fixture(scope="module", params=PRODUCTS)
def key(request):
    return request.param


@pytest.fixture(scope="module")
def zip_path(key, tmp_path_factory):
    return _build_zip(key, tmp_path_factory)


@pytest.fixture(scope="module")
def dir_product(key):
    path = PRODUCT_PATHS[key]
    if not path.exists():
        pytest.skip(f"test product missing: {path}")
    return satimg.open(str(path))


@pytest.fixture(scope="module")
def zip_product(zip_path):
    with satimg.open(zip_path) as p:
        yield p


def test_timestamp_matches(dir_product, zip_product):
    assert zip_product.timestamp == dir_product.timestamp


def test_footprint_matches(dir_product, zip_product):
    assert zip_product.footprint == dir_product.footprint


def test_transformer_matches(dir_product, zip_product):
    h, w = dir_product.height, dir_product.width
    points = [(0, 0), (h // 2, w // 2), (h - 1, w - 1)]
    a = zip_product.transformer.rowcol_to_latlon(points)
    b = dir_product.transformer.rowcol_to_latlon(points)
    numpy.testing.assert_allclose(a, b, atol=1e-6)


def test_thumbnail_matches(dir_product, zip_product):
    assert zip_product.thumbnail().size == dir_product.thumbnail().size


def test_metadata_matches(dir_product, zip_product):
    zm, dm = zip_product.metadata, dir_product.metadata
    assert zm.attrs == dm.attrs
    for name in dm.fields:
        numpy.testing.assert_allclose(
            getattr(zm, name)._values, getattr(dm, name)._values
        )


@pytest.mark.parametrize("what", ["raw", "visual"])
def test_raster_views_match(dir_product, zip_product, what):
    a, b = getattr(zip_product, what), getattr(dir_product, what)
    assert a.shape == b.shape and a.dtype == b.dtype
    assert list(a.band.values) == list(b.band.values)
    window = dict(y=slice(a.sizes["y"] // 2, a.sizes["y"] // 2 + 64),
                  x=slice(a.sizes["x"] // 2, a.sizes["x"] // 2 + 64))
    numpy.testing.assert_array_equal(a.isel(**window).values, b.isel(**window).values)


def test_patches_match(dir_product, zip_product):
    a = next(iter(zip_product.patches_at([(100, 100)], 64)))
    b = next(iter(dir_product.patches_at([(100, 100)], 64)))
    assert a.window == b.window
    numpy.testing.assert_array_equal(a.raw.values, b.raw.values)
    numpy.testing.assert_array_equal(a.visual.values, b.visual.values)


def test_deflated_archive_reads_the_same_pixels(dir_product, key, tmp_path_factory):
    with satimg.open(_build_zip(key, tmp_path_factory, zipfile.ZIP_DEFLATED)) as product:
        patch = next(iter(product.patches_at([(100, 100)], 64)))
    expected = next(iter(dir_product.patches_at([(100, 100)], 64)))
    numpy.testing.assert_array_equal(patch.raw.values, expected.raw.values)


def test_raster_path_points_into_the_archive(zip_path):
    with satimg.open(zip_path) as product:
        path = product._source.raster_path("measurement/band.tiff")
    root = product._source._root
    assert path == f"/vsizip/{{{os.path.abspath(zip_path)}}}/{root}/measurement/band.tiff"
