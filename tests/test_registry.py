import pytest

import satimg
from satimg.products import LandsatProduct, Sentinel1Product, Sentinel2L1CProduct
from satimg.registry import UnknownProductError, resolve


@pytest.mark.parametrize(
    "name, expected",
    [
        ("S1A_IW_GRDH_1SDV_20210102T224803_20210102T224828_035965_04368F_4953_COG.SAFE", Sentinel1Product),
        ("S1A_EW_GRDM_1SDH_20240215T195548_20240215T195633_052574_065C0C_17E5_COG.SAFE", Sentinel1Product),
        ("S2A_MSIL1C_20220114T103401_N0301_R108_T33UUB_20220114T123457.SAFE", Sentinel2L1CProduct),
        ("LC09_L1TP_194022_20251013_20251013_02_T1", LandsatProduct),
        ("/some/dir/LC08_L1GT_100200_20200101_20200101_02_T1", LandsatProduct),
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
