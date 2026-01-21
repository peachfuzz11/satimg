import os
import unittest

from satproducts.products.landsat.landsat_path_handler import LandsatPathHandler
from satproducts.products.landsat.landsat_product import LandsatProduct
from satproducts.sat_product import SatProduct
from tests.test_helper import BASE_DIR


class LandsatHandlerTest(unittest.TestCase):
    PRODUCT_PATH = os.path.join(BASE_DIR, 'data', 'products_minified', "LC09_L1TP_194022_20251013_20251013_02_T1")
    PRODUCT = LandsatProduct
    HANDLER = LandsatPathHandler

    def test_path_handler_matches(self):
        matches = self.HANDLER().matches(self.PRODUCT_PATH)
        self.assertTrue(matches)

    def test_path_handler_fail(self):
        matches = self.HANDLER().matches("1289312oj3i12j321oij23oi12")
        self.assertFalse(matches)

    def test_sat_product(self):
        satproduct = SatProduct.from_path(self.PRODUCT_PATH)
        self.assertTrue(isinstance(satproduct.get_product(), self.PRODUCT))
