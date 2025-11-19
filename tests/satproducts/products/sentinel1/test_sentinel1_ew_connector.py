import datetime
import os
import unittest
from pathlib import Path

from satproducts.products.sentinel.sentinel1.ew.sentinel1_ew_connector import Sentinel1EWConnector


class Sentinel1EWConnectorTest(unittest.TestCase):
    PRODUCT_ID = "S1A_EW_GRDM_1SDH_20251013T074932_20251013T075032_061404_07A9EC_83B9_COG"
    CONNECTOR = Sentinel1EWConnector
    geojson = {
        "type": "Polygon",
        "coordinates": [
            [
                [-14.0, 64.5],
                [-14.7, 64.5],
                [-14.7, 66.8],
                [-14.0, 66.8],
                [-14.0, 64.5]
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
