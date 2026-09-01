import os

import pytest

from tests.test_helper import BASE_DIR

_PRODUCTS = BASE_DIR / "data" / "products_minified"

PRODUCT_PATHS = {
    "sentinel1_iw": _PRODUCTS
    / "S1A_IW_GRDH_1SDV_20210102T224803_20210102T224828_035965_04368F_4953_COG.SAFE",
    "sentinel1_ew": _PRODUCTS
    / "S1A_EW_GRDM_1SDH_20240215T195548_20240215T195633_052574_065C0C_17E5_COG.SAFE",
    "sentinel2_l1c": _PRODUCTS
    / "S2A_MSIL1C_20220114T103401_N0301_R108_T33UUB_20220114T123457.SAFE",
    "landsat": _PRODUCTS / "LC09_L1TP_194022_20251013_20251013_02_T1",
}


def _product(key):
    import satimg

    path = PRODUCT_PATHS[key]
    if not os.path.exists(path):
        pytest.skip(f"test product missing: {path}")
    return satimg.open(str(path))


@pytest.fixture(scope="module")
def sentinel1_iw():
    return _product("sentinel1_iw")


@pytest.fixture(scope="module")
def sentinel2_l1c():
    return _product("sentinel2_l1c")


@pytest.fixture(scope="module")
def landsat():
    return _product("landsat")
