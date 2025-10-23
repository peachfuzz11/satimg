import os

from satproducts.products.sentinel.sentinel1.sentinel1_ew_product import Sentinel1EWProduct
from tests.satproducts.products.product_test import ProductTest
from tests.test_helper import BASE_DIR


class Sentinel1EWProductTest(ProductTest):
    PRODUCT_PATH = os.path.join(BASE_DIR, 'data', 'products_minified',
                                'S1A_EW_GRDM_1SDH_20240215T195548_20240215T195633_052574_065C0C_17E5_COG.SAFE')
    PRODUCT = Sentinel1EWProduct
