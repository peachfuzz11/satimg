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
