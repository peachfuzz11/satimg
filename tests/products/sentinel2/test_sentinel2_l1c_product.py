import os
import unittest

from products.sentinel.sentinel2.sentinel2_l1c_product import Sentinel2L1CProduct
from tests.products.read_test import ReadTest
from tests.products.transform_test import TransformTest
from tests.test_helper import BASE_DIR


class Sentinel2L1CProductTest(ReadTest, TransformTest, unittest.TestCase):
    PRODUCT_PATH = os.path.join(BASE_DIR, 'data', 'products_minified',
                                'S2A_MSIL1C_20220114T103401_N0301_R108_T33UUB_20220114T123457.SAFE')
    PRODUCT = Sentinel2L1CProduct
