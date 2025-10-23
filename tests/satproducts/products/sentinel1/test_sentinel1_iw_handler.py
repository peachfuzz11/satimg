import os

from satproducts.products.sentinel.sentinel1.iw.sentinel1_iw_path_handler import Sentinel1IWPathHandler
from satproducts.products.sentinel.sentinel1.iw.sentinel1_iw_product import Sentinel1IWProduct
from tests.satproducts.products.handler_test import HandlerTest
from tests.test_helper import BASE_DIR


class Sentinel1IWHandlerTest(HandlerTest):
    PRODUCT_PATH = os.path.join(BASE_DIR, 'data', 'products_minified',
                                'S1A_IW_GRDH_1SDV_20210102T224803_20210102T224828_035965_04368F_4953_COG.SAFE')
    PRODUCT = Sentinel1IWProduct
    HANDLER = Sentinel1IWPathHandler
