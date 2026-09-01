from satproducts.products.landsat.landsat_path_handler import LandsatPathHandler
from satproducts.products.sat_product_type import LANDSAT, SENTINEL_1_EW, SENTINEL_1_IW, SENTINEL_2_L1C
from satproducts.products.sentinel.sentinel1.ew.sentinel1_ew_path_handler import Sentinel1EWPathHandler
from satproducts.products.sentinel.sentinel1.iw.sentinel1_iw_path_handler import Sentinel1IWPathHandler
from satproducts.products.sentinel.sentinel2.sentinel2_l1c_path_handler import Sentinel2L1CPathHandler


PATTERN_HANDLERS = [
    (Sentinel1IWPathHandler, SENTINEL_1_IW),
    (Sentinel1EWPathHandler, SENTINEL_1_EW),
    (Sentinel2L1CPathHandler, SENTINEL_2_L1C),
    (LandsatPathHandler, LANDSAT),
]


def resolve_product_type(path: str) -> str:
    for handler, product_type in PATTERN_HANDLERS:
        if handler().matches(path):
            return product_type
    raise NotImplementedError(path)