"""``satimg.open(zip_path)`` (a zip archive, detected by content) reads a
product's metadata straight off the archive, with nothing extracted to disk
-- compared here against the same product opened normally (directory-
backed), not golden values. Pixel access stays extraction-only (use
``satimg.open_zip`` instead) and must raise clearly."""

import numpy
import pytest

import satimg
from tests.conftest import PRODUCT_PATHS
from tests.test_resources import _build_zip

PRODUCTS = ["sentinel1_iw", "sentinel2_l1c", "landsat"]

#: Landsat's per-pixel angle metadata has no XML/JSON equivalent -- it's the
#: one deliberate exclusion from zip-native mode (see product.py).
_METADATA_SUPPORTED = {"sentinel1_iw", "sentinel2_l1c"}


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


def test_metadata_matches_or_raises(key, dir_product, zip_product):
    if key not in _METADATA_SUPPORTED:
        with pytest.raises(satimg.ZipNativeUnsupportedError):
            zip_product.metadata
        return

    zm, dm = zip_product.metadata, dir_product.metadata
    assert zm.attrs == dm.attrs
    name = next(iter(dm.fields))
    numpy.testing.assert_allclose(
        getattr(zm, name)._values, getattr(dm, name)._values
    )


@pytest.mark.parametrize("what", ["raw", "visual"])
def test_raster_views_raise(zip_product, what):
    with pytest.raises(satimg.ZipNativeUnsupportedError):
        getattr(zip_product, what)


def test_patches_raises(zip_product):
    with pytest.raises(satimg.ZipNativeUnsupportedError):
        next(iter(zip_product.patches(64)))


def test_patches_at_raises(zip_product):
    with pytest.raises(satimg.ZipNativeUnsupportedError):
        next(iter(zip_product.patches_at([(0, 0)], 64)))
