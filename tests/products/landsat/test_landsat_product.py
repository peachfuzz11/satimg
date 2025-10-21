import os
import unittest

from satproducts.products.landsat.landsat_handler import LandsatHandler
from satproducts.products.landsat.landsat_product import LandsatProduct
from tests.products.handler_test import HandlerTest
from tests.products.read_test import ReadTest
from tests.products.transform_test import TransformTest
from tests.test_helper import BASE_DIR


class LandsatProductTest(ReadTest, TransformTest,HandlerTest, unittest.TestCase):
    PRODUCT_PATH = os.path.join(BASE_DIR, 'data', 'products_minified', "LC09_L1TP_194022_20251013_20251013_02_T1")
    PRODUCT = LandsatProduct
    HANDLER = LandsatHandler
