"""``s2_utils.parse_tl`` on ``MTD_TL.xml`` variants made from the minified
Sentinel-2 test product's own file."""

import re

import numpy
import pytest

from satimg import s2_utils
from satimg.metadata import Field
from satimg.source import DirSource
from tests.conftest import PRODUCT_PATHS

_MTD_TL = next((PRODUCT_PATHS["sentinel2_l1c"] / "GRANULE").glob("*/MTD_TL.xml"))
_VIEW_GRID = re.compile(r"(<Viewing_Incidence_Angles_Grids\b.*?</Viewing_Incidence_Angles_Grids>)", re.S)
_VALUES_LIST = re.compile(r"<Values_List>.*?</Values_List>", re.S)
_TINY_VALUES = "<Values_List><VALUES>7.1</VALUES><VALUES>7.2</VALUES></Values_List>"  # 2 x 1
_STEP_PX = s2_utils.ANGLE_STEP_M / 10  # 10 m pixels


@pytest.fixture
def mtd_tl_text():
    if not _MTD_TL.exists():
        pytest.skip(f"test product missing: {_MTD_TL}")
    return _MTD_TL.read_text()


def _parse(tmp_path, text):
    (tmp_path / "MTD_TL.xml").write_text(text)
    return s2_utils.parse_tl(DirSource(str(tmp_path)))


def _shrink_view_grids(text, count=None):
    """Replace the Values_Lists of the first ``count`` (all if None) detector
    grids with a 2 x 1 one."""
    seen = 0

    def shrink(match):
        nonlocal seen
        seen += 1
        if count is not None and seen > count:
            return match.group(1)
        return _VALUES_LIST.sub(_TINY_VALUES, match.group(1))

    return _VIEW_GRID.sub(shrink, text)


def _fields(tl):
    return {name: Field(*tl["axes"][name], grid, name=name) for name, grid in tl["grids"].items()}


def test_full_grids_share_the_sun_grids_axes(tmp_path, mtd_tl_text):
    tl = _parse(tmp_path, mtd_tl_text)

    rows, cols = tl["axes"]["sun_zenith"]
    for name in tl["grids"]:
        numpy.testing.assert_array_equal(tl["axes"][name][0], rows)
        numpy.testing.assert_array_equal(tl["axes"][name][1], cols)
    assert not numpy.isnan(tl["grids"]["view_zenith"]).any()


def test_small_viewing_grids_keep_their_own_shape(tmp_path, mtd_tl_text):
    tl = _parse(tmp_path, _shrink_view_grids(mtd_tl_text))

    fields = _fields(tl)  # used to raise "values (2, 1) do not match grid (23, 23)"
    assert tl["grids"]["sun_zenith"].shape == (23, 23)
    assert tl["grids"]["view_zenith"].shape == (2, 1)
    numpy.testing.assert_array_equal(tl["axes"]["view_zenith"][0], [0, _STEP_PX])
    numpy.testing.assert_array_equal(tl["axes"]["view_zenith"][1], [0])
    assert fields["view_zenith"].at((0, 0)) == pytest.approx(7.1)
    assert fields["view_zenith"].at((_STEP_PX, 0)) == pytest.approx(7.2)
    assert fields["view_zenith"].at((_STEP_PX / 2, 5 * _STEP_PX)) == pytest.approx(7.15)


def test_detector_grids_of_different_shapes_are_merged_from_the_corner(tmp_path, mtd_tl_text):
    tl = _parse(tmp_path, _shrink_view_grids(mtd_tl_text, count=1))

    _fields(tl)
    assert tl["grids"]["view_zenith"].shape == (23, 23)
    assert not numpy.isnan(tl["grids"]["view_zenith"]).any()


def test_mean_grids_pads_smaller_grids_with_nan():
    small = numpy.array([[1.0], [3.0]])
    large = numpy.array([[3.0, 5.0], [5.0, 7.0]])

    merged = s2_utils._mean_grids([small, large], circular=False)

    numpy.testing.assert_array_equal(merged, [[2.0, 5.0], [4.0, 7.0]])
