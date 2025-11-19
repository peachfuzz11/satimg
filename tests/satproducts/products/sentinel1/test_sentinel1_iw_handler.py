import os
import unittest

from satproducts.products.sentinel.sentinel1.iw.sentinel1_iw_path_handler import Sentinel1IWPathHandler
from satproducts.products.sentinel.sentinel1.iw.sentinel1_iw_product import Sentinel1IWProduct
from satproducts.sat_products import SatProducts
from tests.test_helper import BASE_DIR


class Sentinel1IWHandlerTest(unittest.TestCase):
    PRODUCT_PATH = os.path.join(BASE_DIR, 'data', 'products_minified',
                                'S1A_IW_GRDH_1SDV_20210102T224803_20210102T224828_035965_04368F_4953_COG.SAFE')
    PRODUCT = Sentinel1IWProduct
    HANDLER = Sentinel1IWPathHandler

    def test_path_handler_matches(self):
        matches = self.HANDLER().matches(self.PRODUCT_PATH)
        self.assertTrue(matches)

    def test_path_handler_fail(self):
        matches = self.HANDLER().matches("1289312oj3i12j321oij23oi12")
        self.assertFalse(matches)

    def test_sat_product(self):
        satproduct = SatProducts.from_path(self.PRODUCT_PATH)
        self.assertTrue(isinstance(satproduct.get_product(), self.PRODUCT))
