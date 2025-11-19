import datetime
import os
import unittest
from pathlib import Path

from satproducts.products.sentinel.sentinel1.iw.sentinel1_iw_connector import Sentinel1IWConnector


class Sentinel1IWConnectorTest(unittest.TestCase):
    PRODUCT_ID = "S1C_IW_GRDH_1SDV_20251013T052401_20251013T052426_004539_008FB4_D25E_COG"
    CONNECTOR = Sentinel1IWConnector

    geojson = {
        "type": "Polygon",
        "coordinates": [
            [
                [10.0, 54.5],
                [12.7, 54.5],
                [12.7, 56.8],
                [10.0, 56.8],
                [10.0, 54.5]
            ]
        ]
    }
    start = datetime.datetime(2025, 10, 10)
    end = datetime.datetime(2025, 10, 14)

    save_path = os.path.join(Path(__file__).parent.parent, "tmp")
    os.makedirs(save_path, exist_ok=True)

    def test_search(self):
        items = self.CONNECTOR().search(geojson=self.geojson, start_datetime=self.start, stop_datetime=self.end)
        print(len(items))

    def test_download(self):
        connector = self.CONNECTOR()
        # items = connector.search(geojson=self.geojson, start_datetime=self.start, stop_datetime=self.end)
        # with TemporaryDirectory(dir=self.save_path) as tmpdir:
        # zippath = connector.download(items[0], save_path=os.path.join(self.save_path, f"{items[0]["id"]}.zip"))
        # print(zippath)

    def test_search_by_id(self):
        items = self.CONNECTOR().search(ids=[self.PRODUCT_ID])
        print(items)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["id"], self.PRODUCT_ID)
