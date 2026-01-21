import os
import unittest

from satproducts.products.sentinel.sentinel1.ew.sentinel1_ew_path_handler import Sentinel1EWPathHandler
from satproducts.products.sentinel.sentinel1.ew.sentinel1_ew_product import Sentinel1EWProduct
from satproducts.sat_product import SatProduct
from tests.test_helper import BASE_DIR


class Sentinel1EWHandlerTest(unittest.TestCase):
    PRODUCT_PATH = os.path.join(BASE_DIR, 'data', 'products_minified',
                                'S1A_EW_GRDM_1SDH_20240215T195548_20240215T195633_052574_065C0C_17E5_COG.SAFE')
    PRODUCT = Sentinel1EWProduct
    HANDLER = Sentinel1EWPathHandler

    def test_path_handler_matches(self):
        matches = self.HANDLER().matches(self.PRODUCT_PATH)
        self.assertTrue(matches)

    def test_path_handler_fail(self):
        matches = self.HANDLER().matches("1289312oj3i12j321oij23oi12")
        self.assertFalse(matches)

    def test_sat_product(self):
        satproduct = SatProduct.from_path(self.PRODUCT_PATH)
        self.assertTrue(isinstance(satproduct.get_product(), self.PRODUCT))
