import os
import unittest

from satproducts.products.sentinel.sentinel1.sentinel1_ew_product import Sentinel1EWProduct
from tests.products.read_test import ReadTest
from tests.products.transform_test import TransformTest
from tests.test_helper import BASE_DIR


class Sentinel1EWProductTest(ReadTest, TransformTest, unittest.TestCase):
    PRODUCT_PATH = os.path.join(BASE_DIR, 'data', 'products_minified',
                                'S1A_EW_GRDM_1SDH_20240215T195548_20240215T195633_052574_065C0C_17E5_COG.SAFE')
    PRODUCT = Sentinel1EWProduct
