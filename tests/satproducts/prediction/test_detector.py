import os
import unittest

from satproducts.prediction.detector import Detector
from satproducts.products.sentinel.sentinel2.sentinel2_l1c_product import Sentinel2L1CProduct
from tests.test_helper import BASE_DIR


class TestDetector(unittest.TestCase):
    PRODUCT_PATH = os.path.join(BASE_DIR, 'data', 'products_minified',
                                'S2A_MSIL1C_20220114T103401_N0301_R108_T33UUB_20220114T123457.SAFE')

    def test_detection(self):
        detector = Detector(Sentinel2L1CProduct(self.PRODUCT_PATH))
        detections = detector.detect()
        print(detections)
