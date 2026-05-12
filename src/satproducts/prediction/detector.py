import typing

import numpy
from roaring_landmask.roaring_landmask import RoaringLandmask

from satproducts.prediction import nms
from satproducts.prediction.dto import BBox, Label, Detection, Coordinate
from satproducts.prediction.models.model import Model
from satproducts.products.base.product import Product


class Detector:

    def __init__(self, product: Product, model: Model, **kwargs):
        self._product = product
        self._model = model
        self._land_mask = RoaringLandmask.new()
        self._config = {
            "conf_threshold": 0.1,
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

        for islice, tile in self._product.tile(config.get("slice_size")):
            detections = self._model.predict(tile, **config)
            conf_threshold = config.get("conf_threshold")
            detections = detections[(detections[:, 4] > conf_threshold)]

            if config.get("nms_threshold"):
                keep = nms.non_max_suppression(detections[:, 4:], detections[:, :4],
                                               overlap_threshold=config["nms_threshold"])
                detections = detections[keep]

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
                x, y = x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2
                latlon = self._product.transformer.rowcol_to_latlon((y, x))
                coordinate = Coordinate(lat=float(latlon[0, 0]), lon=float(latlon[0, 1]))
                if config.get("apply_landmask") and self._land_mask.contains(coordinate.lon, coordinate.lat):
                    continue
                detection_dto = Detection(bbox, label, coordinate)
                detection_list.append(detection_dto)
        return detection_list
