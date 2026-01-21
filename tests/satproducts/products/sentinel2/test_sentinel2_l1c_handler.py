import os
import unittest

from satproducts.products.sentinel.sentinel2.sentinel2_l1c_path_handler import Sentinel2L1CPathHandler
from satproducts.products.sentinel.sentinel2.sentinel2_l1c_product import Sentinel2L1CProduct
from satproducts.sat_product import SatProduct
from tests.test_helper import BASE_DIR


class Sentinel2L1CProductTest(unittest.TestCase):
    PRODUCT_PATH = os.path.join(BASE_DIR, 'data', 'products_minified',
                                'S2A_MSIL1C_20220114T103401_N0301_R108_T33UUB_20220114T123457.SAFE')
    PRODUCT = Sentinel2L1CProduct
    HANDLER = Sentinel2L1CPathHandler

    def test_path_handler_matches(self):
        matches = self.HANDLER().matches(self.PRODUCT_PATH)
        self.assertTrue(matches)

    def test_path_handler_fail(self):
        matches = self.HANDLER().matches("1289312oj3i12j321oij23oi12")
        self.assertFalse(matches)

    def test_sat_product(self):
        satproduct = SatProduct.from_path(self.PRODUCT_PATH)
        self.assertTrue(isinstance(satproduct.get_product(), self.PRODUCT))
