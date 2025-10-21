import os
import unittest

from satproducts.products.landsat.landsat_handler import LandsatHandler
from satproducts.products.sentinel.sentinel1.sentinel1_ew_product import Sentinel1EWProduct
from satproducts.products.sentinel.sentinel1.sentinel1_iw_product import Sentinel1IWProduct
from satproducts.products.sentinel.sentinel2.sentinel2_l1c_product import Sentinel2L1CProduct
from satproducts.products.sentinel.sentinel_handler import SentinelHandler
from satproducts.sat_product_factory import SatProductFactory
from tests.test_helper import BASE_DIR


class HandlerTest(unittest.TestCase):
    all_handlers = [SentinelHandler(), LandsatHandler()]

    def test_fail_handler(self):
        path = ""
        for handler in self.all_handlers:
            product = handler.create_product(product_path=path)
            self.assertTrue(product is None)
            try:
                product = handler.handle(path)
            except Exception as e:
                pass
            self.assertTrue(product is None)

    def test_fail_factory(self):
        path = ""
        product = None
        try:
            product = SatProductFactory(path).create()
        except Exception as e:
            pass
        self.assertTrue(product is None)

    def test_handlers(self):
        for data in self.test_data:
            handler = data["handler"]
            path = data["product_path"]
            product = data["product"]
            result = handler().handle(product_path=path)
            self.assertTrue(product is not None)
            self.assertTrue(isinstance(result, product))

    def test_factory(self):
        for data in self.test_data:
            path = data["product_path"]
            product = data["product"]
            result = SatProductFactory(product_path=path).create()
            self.assertTrue(product is not None)
            self.assertTrue(isinstance(result, product))

    def test_init(self):
        for data in self.test_data:
            path = data["product_path"]
            product = data["product"]
            result = SatProductFactory(product_path=path).create()
            self.assertTrue(product is not None)
            self.assertTrue(isinstance(result, product))

    DATA = [
        (SentinelHandler, "S1A_EW_GRDM_1SDH_20240215T195548_20240215T195633_052574_065C0C_17E5_COG.SAFE",
         Sentinel1EWProduct, 2),
        (SentinelHandler, "S1A_IW_GRDH_1SDH_20141006T074149_20141006T074214_002706_00306E_51E0_COG.SAFE",
         Sentinel1IWProduct, 2),
        (SentinelHandler, "S1A_IW_GRDH_1SDV_20210102T224803_20210102T224828_035965_04368F_4953_COG.SAFE",
         Sentinel1IWProduct, 2),
        (SentinelHandler, "S2A_MSIL1C_20220114T103401_N0301_R108_T33UUB_20220114T123457.SAFE", Sentinel2L1CProduct, 2),
    ]

    test_data = []
    for handler, product_path, product, channels in DATA:
        test_data.append(
            {"handler": handler, "product_path": os.path.join(BASE_DIR, 'data', 'products_minified', product_path),
             "product": product, "channels": channels})
