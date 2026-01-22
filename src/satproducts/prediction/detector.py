import typing

import numpy
from roaring_landmask.roaring_landmask import RoaringLandmask

from satproducts.prediction.dto import BBox, Label, Detection, Coordinate
from satproducts.prediction.models.model_catalog import ModelCatalog
from satproducts.prediction.models.yolo26_model import Yolo26Model
from satproducts.products.base.product import Product
from satproducts.products.sentinel.sentinel1.iw.sentinel1_iw_product import Sentinel1IWProduct
from satproducts.products.sentinel.sentinel2.sentinel2_l1c_product import Sentinel2L1CProduct


class Detector:

    def __init__(self, product: Product, **kwargs):
        self._product = product
        self._model = detection_factory(product)
        self._land_mask = RoaringLandmask.new()
        self._config = {
            "conf_threshold": 0.25,
            "slice_size": 256,
            "min_size": 1,
            "max_size": 300,
            "min_area": 4,
            "max_area": 300 ** 2,
            "max_aspect_ratio": 15,
            "apply_landmask": True,
        }
        self._config.update(**kwargs)

    def detect(self, *args, **kwargs) -> typing.List[Detection]:
        config = self._config.copy()
        config.update(kwargs)
        detection_list = []
        for islice in self._product.slices(config.get("slice_size")):
            subset = self._product.view(islice)
            detections = self._model.predict(subset, **config)
            conf_threshold = config.get("conf_threshold")
            detections = detections[(detections[:, 4] > conf_threshold)]

            x1 = detections[:, 0]
            y1 = detections[:, 1]
            x2 = detections[:, 2]
            y2 = detections[:, 3]

            w = x2 - x1
            h = y2 - y1
            area = w * h
            aspect = numpy.maximum(w / h, h / w)

            mask = (
                    (config.get("min_size") <= w) &
                    (w <= config.get("max_size")) &
                    (config.get("min_size") <= h) &
                    (h <= config.get("max_size")) &
                    (config.get("min_area") < area) &
                    (area < config.get("max_area")) &
                    (aspect < config.get("max_aspect_ratio"))
            )

            detections = detections[mask]

            detections[:, [0, 2]] += islice.i
            detections[:, [1, 3]] += islice.j
            for detection in detections:
                x1, y1, x2, y2, conf = map(float, detection)
                bbox = BBox(x1, y1, x2, y2)
                label = Label("ship", conf)
                x, y = x1 + x2 / 2, y1 + y2 / 2
                latlon = self._product.transformer.rowcol_to_latlon((y, x))
                coordinate = Coordinate(lat=float(latlon[0, 0]), lon=float(latlon[0, 1]))
                if config.get("apply_landmask") and self._land_mask.contains(coordinate.lon, coordinate.lat):
                    continue
                detection_dto = Detection(bbox, label, coordinate)
                detection_list.append(detection_dto)
        return detection_list


def detection_factory(product: Product, *args, **kwargs) -> Yolo26Model:
    if isinstance(product, Sentinel1IWProduct):
        return Yolo26Model(ModelCatalog.S1.get_path())
    elif isinstance(product, Sentinel2L1CProduct):
        return Yolo26Model(ModelCatalog.S2.get_path())
    else:
        raise NotImplementedError(product)
