import pytest

import satimg
from satimg.products.landsat.c2l1.landsat_c2l1 import LandsatC2L1
from satimg.products.sentinel1.grd.sentinel1_grd import Sentinel1GRD
from satimg.products.sentinel2.l1c.sentinel2_l1c import Sentinel2L1C
from satimg.registry import UnknownProductError, resolve


@pytest.mark.parametrize(
    "name, expected",
    [
        ("S1A_IW_GRDH_1SDV_20210102T224803_20210102T224828_035965_04368F_4953_COG.SAFE", Sentinel1GRD),
        ("S1A_EW_GRDM_1SDH_20240215T195548_20240215T195633_052574_065C0C_17E5_COG.SAFE", Sentinel1GRD),
        ("S2A_MSIL1C_20220114T103401_N0301_R108_T33UUB_20220114T123457.SAFE", Sentinel2L1C),
        ("LC09_L1TP_194022_20251013_20251013_02_T1", LandsatC2L1),
        ("/some/dir/LC08_L1GT_100200_20200101_20200101_02_T1", LandsatC2L1),
    ],
)
def test_resolve_by_name(name, expected):
    assert resolve(name) is expected


def test_resolve_unknown():
    with pytest.raises(UnknownProductError):
        resolve("not-a-satellite-product")


def test_open_unknown():
    with pytest.raises(UnknownProductError):
        satimg.open("/tmp/definitely-not-a-product")


_S1 = "S1A_IW_GRDH_1SDV_20210102T224803_20210102T224828_035965_04368F"
_S2 = "S2A_MSIL1C_20220114T103401_{}_R108_T33UUB_{}"
_LANDSAT = "LC09_{}_194022_20251013_{}_02_{}"


@pytest.mark.parametrize(
    "names, key",
    [
        # Sentinel-1: the product unique id, _COG and .SAFE
        (
            [f"{_S1}_4953_COG", f"{_S1}_A1B2_COG", f"{_S1}_4953", f"{_S1}_4953_COG.SAFE",
             f"/data/{_S1}_4953_COG.SAFE.zip"],
            _S1,
        ),
        (
            ["S1A_EW_GRDM_1SDH_20240215T195548_20240215T195633_052574_065C0C_17E5_COG.SAFE"],
            "S1A_EW_GRDM_1SDH_20240215T195548_20240215T195633_052574_065C0C",
        ),
        # Sentinel-2: the processing baseline, the generation time and .SAFE
        (
            [_S2.format("N0301", "20220114T123457"), _S2.format("N0511", "20240301T080000"),
             _S2.format("N0301", "20220114T123457") + ".SAFE"],
            "S2A_MSIL1C_20220114T103401_R108_T33UUB",
        ),
        # Landsat: the correction level, the processing date and the tier
        (
            [_LANDSAT.format("L1TP", "20251013", "T1"), _LANDSAT.format("L1GT", "20251101", "T2")],
            "LC09_194022_20251013",
        ),
        # anything else is its own scene
        (["not-a-satellite-product"], "not-a-satellite-product"),
    ],
)
def test_scene_key_is_shared_by_every_version_of_a_scene(names, key):
    assert {satimg.scene_key(name) for name in names} == {key}


@pytest.mark.parametrize(
    "a, b",
    [
        (f"{_S1}_4953_COG", f"{_S1}_4953_COG".replace("04368F", "04368E")),  # datatake
        (_S2.format("N0301", "20220114T123457"), _S2.format("N0301", "20220114T123457").replace("T33UUB", "T33UUC")),
        (_S2.format("N0301", "20220114T123457"), _S2.format("N0301", "20220114T123457").replace("R108", "R008")),
        (_LANDSAT.format("L1TP", "20251013", "T1"), _LANDSAT.format("L1TP", "20251013", "T1").replace("194022", "194023")),
    ],
)
def test_scene_key_tells_scenes_apart(a, b):
    assert satimg.scene_key(a) != satimg.scene_key(b)
