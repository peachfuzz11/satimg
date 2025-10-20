import os
import unittest
from datetime import datetime

from products.sentinel.sentinel2.service.sentinel2_service import get_mtd_msil1c_data, get_mtd_tl_data
from tests.test_helper import BASE_DIR


class TestSentinel2Service(unittest.TestCase):
    file_path = os.path.join(BASE_DIR, 'data', 'products_minified',
                             'S2A_MSIL1C_20220114T103401_N0301_R108_T33UUB_20220114T123457.SAFE')

    def test_get_mtd_msil1c_data(self):
        # Arrange
        test_footprint = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [(13.579708268630132, 55.280642797207044),
                                (13.53512454747758, 55.20776443652352),
                                (13.448200901895712, 55.06504300021859),
                                (13.376784334281517, 54.946938457754705),
                                (11.879220117042458, 54.91902653885059),
                                (11.800511438780461, 55.90416762514414),
                                (13.55602214889276, 55.93727334575647),
                                (13.579708268630132, 55.280642797207044)]
            }
        }
        data = get_mtd_msil1c_data(self.file_path)

        # Act
        bands = data["bands"]
        footprint = data["footprint"]

        # Assert
        self.assertFalse(bands is None)
        self.assertFalse(footprint is None)
        self.assertEqual(footprint, test_footprint)

    def test_get_mtd_tl_data(self):
        # Arrange
        test_timestamp = datetime(2022, 1, 14, 10, 35, 25, 542751)
        data = get_mtd_tl_data(self.file_path)

        # Act
        timestamp = data["timestamp"]

        # Assert
        self.assertFalse(timestamp is None)
        self.assertTrue(isinstance(timestamp, datetime))
        self.assertEqual(timestamp, test_timestamp)
