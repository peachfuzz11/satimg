import os

from satproducts.products.landsat.landsat_path_handler import LandsatPathHandler
from satproducts.products.landsat.landsat_product import LandsatProduct
from tests.satproducts.products.handler_test import HandlerTest
from tests.satproducts.products.product_test import ProductTest
from tests.test_helper import BASE_DIR


class LandsatHandlerTest(HandlerTest):
    PRODUCT_PATH = os.path.join(BASE_DIR, 'data', 'products_minified', "LC09_L1TP_194022_20251013_20251013_02_T1")
    PRODUCT = LandsatProduct
    HANDLER = LandsatPathHandler
