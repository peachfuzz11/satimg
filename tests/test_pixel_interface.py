"""Every product reads its pixels through the same lazy interface: a
``BandStack`` of ``Raster`` s, opened, chunked, decimated and closed by
``Product`` itself -- whatever the file format behind it."""

import os

import numpy
import pytest

import satimg
from satimg.readers.raster import Raster
from satimg.readers.stack import BandStack
from satimg.readers.thumbnail import THUMBNAIL_SIZE
from tests import modis_fixture, nisar_fixture
from tests.conftest import _PRODUCTS, PRODUCT_PATHS

_OTHERS = {
    "viirs": _PRODUCTS / "VJ102DNB.A2026305.1630.021.2026305232544",
    "sentinel3": _PRODUCTS
    / "S3A_OL_1_EFR____20240916T101050_20240916T101350_20240917T104150_0179_117_065_1800_PS1_O_NT_004.SEN3",
}
KINDS = ["sentinel1_iw", "sentinel2_l1c", "landsat", "viirs", "sentinel3", "modis", "nisar"]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """The synthetic products, built once."""
    d = tmp_path_factory.mktemp("synthetic")
    modis_fixture.build(d / modis_fixture.TERRA)
    modis_fixture.build_geolocation(d / modis_fixture.TERRA_GEO)
    nisar_fixture.build(d / nisar_fixture.NAME)
    return {"modis": d / modis_fixture.TERRA, "nisar": d / nisar_fixture.NAME}


@pytest.fixture(params=KINDS)
def product(request, built):
    path = PRODUCT_PATHS.get(request.param) or _OTHERS.get(request.param) or built[request.param]
    if not os.path.exists(path):
        pytest.skip(f"test product missing: {path}")
    with satimg.open(str(path)) as p:
        yield p


def test_bands_are_a_stack_of_rasters(product):
    stack = product._raw_bands()
    assert isinstance(stack, BandStack)
    assert stack.rasters and all(isinstance(r, Raster) for r in stack.rasters)


def test_raw_opens_lazily_chunked_to_the_tile_and_closes(product):
    stack = product._raw_bands()
    raw = product._open_raw(128)
    try:
        assert raw.dims == ("band", "y", "x")
        assert raw.chunks is not None, "raw must be dask-backed, i.e. lazy"
        assert max(raw.chunks[1]) <= 128 and max(raw.chunks[2]) <= 128
        assert list(raw.band.values) == list(stack.names)
        assert (raw.sizes["y"], raw.sizes["x"]) == (product.height, product.width)
    finally:
        raw.close()


def test_visual_is_lazy_uint8_on_the_raw_grid(product):
    visual = product._open_visual(128)
    try:
        assert visual.dims == ("band", "y", "x") and visual.dtype == numpy.uint8
        assert visual.chunks is not None
        assert (visual.sizes["y"], visual.sizes["x"]) == (product.height, product.width)
    finally:
        visual.close()


def test_overviews_are_the_rasters_own(product):
    factors = product.visual_overviews
    assert factors[0] == 1 and factors == sorted(factors)
    for factor in factors[1:]:
        assert product.visual_overview(factor).sizes["x"] < product.width
    with pytest.raises(ValueError):
        product.visual_overview(factors[-1] * 3 + 1)


def test_thumbnail_is_thumbnail_sized(product):
    assert product.thumbnail().size == THUMBNAIL_SIZE
