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
