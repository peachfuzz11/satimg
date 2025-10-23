import os

from satproducts.products.sentinel.sentinel1.sentinel1_iw_product import Sentinel1IWProduct
from tests.satproducts.products.product_test import ProductTest
from tests.test_helper import BASE_DIR


class Sentinel1IWProductTest(ProductTest):
    PRODUCT_PATH = os.path.join(BASE_DIR, 'data', 'products_minified',
                                'S1A_IW_GRDH_1SDV_20210102T224803_20210102T224828_035965_04368F_4953_COG.SAFE')
    PRODUCT = Sentinel1IWProduct
