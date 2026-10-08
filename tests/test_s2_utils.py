"""``s2_utils.parse_tl`` on ``MTD_TL.xml`` variants made from the minified
Sentinel-2 test product's own file."""

import re
import shutil

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
    rows, cols = tl["axes"]
    return {name: Field(rows, cols, grid, name=name) for name, grid in tl["grids"].items()}


def test_full_grids_are_unchanged(tmp_path, mtd_tl_text):
    tl = _parse(tmp_path, mtd_tl_text)
    shape = tl["grids"]["sun_zenith"].shape
    for name, grid in tl["grids"].items():
        assert grid.shape == shape, name
    assert not numpy.isnan(tl["grids"]["view_zenith"]).any()


def test_unplaceable_detector_grids_are_dropped(tmp_path, mtd_tl_text, caplog):
    full = _parse(tmp_path, mtd_tl_text)
    tl = _parse(tmp_path, _shrink_view_grids(mtd_tl_text, count=1))

    _fields(tl)  # every field fits the shared axes again
    assert tl["grids"]["view_zenith"].shape == full["grids"]["sun_zenith"].shape
    assert "dropping 1 of" in caplog.text


def test_no_placeable_detector_grid_falls_back_to_the_mean_viewing_angle(tmp_path, mtd_tl_text):
    tl = _parse(tmp_path, _shrink_view_grids(mtd_tl_text))

    fields = _fields(tl)  # used to raise "values (2, 1) do not match grid (23, 23)"
    zenith = tl["grids"]["view_zenith"]
    assert zenith.shape == tl["grids"]["sun_zenith"].shape
    assert numpy.all(zenith == zenith[0, 0])
    assert 0 < zenith[0, 0] < 15  # near-nadir, from Mean_Viewing_Incidence_Angle_List
    assert 0 <= fields["view_azimuth"].at((0, 0)) < 360


def test_no_mean_viewing_angles_leaves_nan(tmp_path, mtd_tl_text):
    text = re.sub(r"<Mean_Viewing_Incidence_Angle_List>.*?</Mean_Viewing_Incidence_Angle_List>", "",
                  _shrink_view_grids(mtd_tl_text), flags=re.S)
    tl = _parse(tmp_path, text)

    assert numpy.isnan(tl["grids"]["view_zenith"]).all()
