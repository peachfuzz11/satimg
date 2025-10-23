import abc
import unittest

from satproducts.sat_product import SatProduct


class HandlerTest(abc.ABC, unittest.TestCase):
    PRODUCT_PATH = None
    PRODUCT = None
    HANDLER = None

    def skip(self):
        if self.PRODUCT_PATH is None or self.PRODUCT is None or self.HANDLER is None:
            self.skipTest("ABC")

    def test_path_handler_matches(self):
        self.skip()
        matches = self.HANDLER().matches(self.PRODUCT_PATH)
        self.assertTrue(matches)

    def test_path_handler_fail(self):
        self.skip()
        matches = self.HANDLER().matches("1289312oj3i12j321oij23oi12")
        self.assertFalse(matches)

    def test_sat_product(self):
        self.skip()
        satproduct = SatProduct(self.PRODUCT_PATH)
        print(satproduct.get_product(), self.PRODUCT)
        self.assertTrue(isinstance(satproduct.get_product(), self.PRODUCT))
