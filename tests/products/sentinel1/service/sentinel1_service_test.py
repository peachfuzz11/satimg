import datetime
import os
import unittest

from satproducts.products.sentinel.sentinel1.service.sentinel1_service import get_manifest_data
from tests.test_helper import BASE_DIR


class Sentinel1ServiceTest(unittest.TestCase):
    file_path = os.path.join(BASE_DIR, 'data', 'products_minified',
                             'S1A_IW_GRDH_1SDH_20141006T074149_20141006T074214_002706_00306E_51E0_COG.SAFE')

    def test_get_manifest_data(self):
        # Arrange
        test_start_time = datetime(2014, 10, 6, 7, 41, 49, 149854)
        test_stop_time = datetime(2014, 10, 6, 7, 42, 14, 148815)
        test_timestamp = (test_stop_time - test_start_time) / 2 + test_start_time
        test_footprint = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [(-17.001230, 62.042053),
                                (-21.795715, 62.474052),
                                (-21.271393, 63.962234),
                                (-16.232010, 63.519577)]
            }
        }

        # Act
        data = get_manifest_data(self.file_path)
        timestamp = data["timestamp"]
        footprint = data["footprint"]

        # Assert
        self.assertFalse(timestamp is None)
        self.assertEqual(timestamp, test_timestamp)
        self.assertFalse(footprint is None)
        self.assertEqual(footprint, test_footprint)
