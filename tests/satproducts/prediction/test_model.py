import os
import unittest

import PIL.Image

from satproducts.prediction.models.model_catalog import ModelCatalog
from satproducts.prediction.models.yolo26_model import Yolo26Model
from tests.test_helper import BASE_DIR


class TestModel(unittest.TestCase):
    IMG_PATH = os.path.join(BASE_DIR, 'data', 'imgs')

    def test_detection(self):
        model = Yolo26Model(ModelCatalog.S2.get_path())
        imgs = [os.path.join(self.IMG_PATH, i) for i in os.listdir(self.IMG_PATH)]
        for img_path in imgs:
            with PIL.Image.open(img_path) as img:
                detections = model.predict(img)
                print(detections)

    def test_landmask(self):
        from roaring_landmask.roaring_landmask import RoaringLandmask
        mask = RoaringLandmask.new()
        cities_coords = [
            (40.7128, -74.0060),  # New York City, USA
            (51.5074, -0.1278),  # London, UK
            (35.6895, 139.6917),  # Tokyo, Japan
            (48.8566, 2.3522),  # Paris, France
            (39.9042, 116.4074),  # Beijing, China
            (55.7558, 37.6173),  # Moscow, Russia
            (-33.8688, 151.2093),  # Sydney, Australia
            (30.0444, 31.2357),  # Cairo, Egypt
            (-22.9068, -43.1729),  # Rio de Janeiro, Brazil
            (43.6532, -79.3832),  # Toronto, Canada
            (19.0760, 72.8777),  # Mumbai, India
            (-26.2041, 28.0473)  # Johannesburg, South Africa
        ]
        for c in cities_coords:
            print(mask.contains(*c[::-1]))
