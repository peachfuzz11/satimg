import typing

from roaring_landmask.roaring_landmask import RoaringLandmask

from satproducts.prediction.dto import BBox, Label, Detection, Coordinate
from satproducts.prediction.models.model_catalog import ModelCatalog
from satproducts.prediction.models.yolo26_model import Yolo26Model
from satproducts.products.base.product import Product
from satproducts.products.sentinel.sentinel1.iw.sentinel1_iw_product import Sentinel1IWProduct
from satproducts.products.sentinel.sentinel2.sentinel2_l1c_product import Sentinel2L1CProduct


class Detector:

    def __init__(self, product: Product):
        self._product = product
        self._model = detection_factory(product)
        self._land_mask = RoaringLandmask.new()

    def detect(self, *args, **kwargs) -> typing.List[Detection]:
        detection_list = []
        slice_size = kwargs.get("slice_size", 256)
        for islice in self._product.slices(slice_size):
            subset = self._product.view(islice)
            detections = self._model.predict(subset, **kwargs)
            detections[:, 0] += islice.i
            detections[:, 1] += islice.j
            for detection in detections:
                x, y, w, h, conf = map(float, detection)
                if not 1 <= w <= 50 or not 1 <= h <= 50:
                    continue
                if not w / h <= 10 or not h / w <= 10:
                    continue
                bbox = BBox(x, y, w, h)
                label = Label("ship", conf)
                latlon = self._product.transformer.rowcol_to_latlon((bbox.y, bbox.x))
                coordinate = Coordinate(lat=float(latlon[0, 0]), lon=float(latlon[0, 1]))
                if not self._land_mask.contains(coordinate.lat, coordinate.lon):
                    continue
                detection_dto = Detection(bbox, label, coordinate)
                detection_list.append(detection_dto)
        return detection_list


def detection_factory(product: Product, *args, **kwargs) -> Yolo26Model:
    if isinstance(product, Sentinel1IWProduct):
        return Yolo26Model(ModelCatalog.S1.get_path(), config={})
    elif isinstance(product, Sentinel2L1CProduct):
        return Yolo26Model(ModelCatalog.S2.get_path())
    else:
        raise NotImplementedError(product)
