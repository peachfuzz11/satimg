"""``Product.visual_overview``: the visual read from the rasters' own
overviews. The minified Sentinel-1 product is blank and ships without
overviews, so a copy gets a block of speckle (``SPECKLE``) and 2x and 4x
overviews written here; Sentinel-2's true-colour image has a 2x one of its
own."""

import shutil
import zipfile
from pathlib import Path

import numpy
import pytest
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

import satimg
from tests.conftest import PRODUCT_PATHS

#: (row, col) of the 1024 px block given speckle
SPECKLE = (4096, 6144)


@pytest.fixture(scope="module")
def s1_path(tmp_path_factory):
    src = PRODUCT_PATHS["sentinel1_iw"]
    if not src.exists():
        pytest.skip(f"test product missing: {src}")
    path = tmp_path_factory.mktemp("overviews") / src.name
    shutil.copytree(src, path)
    rng = numpy.random.default_rng(0)
    for tiff in (path / "measurement").glob("*.tiff"):
        with rasterio.open(tiff, "r+") as dst:
            # speckle over a brightening scene, so there is structure to compare
            scene = numpy.geomspace(10, 3000, 1024)[None, None, :] * rng.gamma(4.0, 0.25, (1, 1024, 1024))
            speckle = scene.astype("uint16")
            dst.write(speckle, window=Window(SPECKLE[1], SPECKLE[0], 1024, 1024))
            dst.build_overviews([2, 4], Resampling.average)
    return path


@pytest.fixture(scope="module")
def s1(s1_path):
    with satimg.open(str(s1_path)) as product:
        yield product


def test_lists_the_overviews_every_visual_raster_has(s1):
    assert s1.visual_overviews == [1, 2, 4]


def test_full_resolution_is_the_visual(s1):
    assert s1.visual_overview(1) is s1.visual


@pytest.mark.parametrize("factor", [2, 4])
def test_overview_is_the_visual_decimated(s1, factor):
    overview = s1.visual_overview(factor)
    visual = s1.visual

    assert overview.dtype == visual.dtype
    assert list(overview.band.values) == list(visual.band.values)
    assert abs(overview.sizes["y"] - visual.sizes["y"] / factor) <= 1
    assert abs(overview.sizes["x"] - visual.sizes["x"] / factor) <= 1
    # the same scene: a window agrees with the visual averaged down to it
    row, col = SPECKLE
    small = overview.isel(y=slice(row // factor, (row + 1024) // factor),
                          x=slice(col // factor, (col + 1024) // factor)).values[0].astype(float)
    full = visual.isel(y=slice(row, row + 1024), x=slice(col, col + 1024)).values[0].astype(float)
    coarse = full.reshape(small.shape[0], factor, small.shape[1], factor).mean(axis=(1, 3))
    assert numpy.corrcoef(small.ravel(), coarse.ravel())[0, 1] > 0.9


def test_an_overview_the_rasters_lack_raises(s1):
    with pytest.raises(ValueError):
        s1.visual_overview(8)


def test_overviews_close_with_the_product(s1_path):
    with satimg.open(str(s1_path)) as product:
        product.visual_overview(2)
        assert product._overviews
    assert not product._overviews


def test_a_zipped_product_reads_the_same_overview(s1, s1_path, tmp_path):
    archive = tmp_path / (s1_path.name + ".zip")
    with zipfile.ZipFile(archive, "w") as z:
        for f in s1_path.rglob("*"):
            if f.is_file():
                z.write(f, arcname=str(Path(s1_path.name) / f.relative_to(s1_path)))

    with satimg.open(str(archive)) as zipped:
        window = dict(y=slice(SPECKLE[0] // 2, SPECKLE[0] // 2 + 512), x=slice(SPECKLE[1] // 2, SPECKLE[1] // 2 + 512))
        numpy.testing.assert_array_equal(zipped.visual_overview(2).isel(**window).values,
                                         s1.visual_overview(2).isel(**window).values)


def test_sentinel2_reads_its_true_colour_images_overview(sentinel2_l1c):
    assert sentinel2_l1c.visual_overviews == [1, 2]
    overview = sentinel2_l1c.visual_overview(2)
    assert list(overview.band.values) == ["red", "green", "blue"]
    assert dict(overview.sizes) == {"band": 3, "y": 5490, "x": 5490}
