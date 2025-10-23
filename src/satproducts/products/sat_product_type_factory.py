from satproducts.products.landsat.landsat_path_handler import LandsatPathHandler
from satproducts.products.sat_product_type import SatProductType
from satproducts.products.sentinel.sentinel1.ew.sentinel1_ew_path_handler import Sentinel1EWPathHandler
from satproducts.products.sentinel.sentinel1.iw.sentinel1_iw_path_handler import Sentinel1IWPathHandler
from satproducts.products.sentinel.sentinel2.sentinel2_l1c_path_handler import Sentinel2L1CPathHandler


class SatProductTypeFactory:

    def __init__(self, sat_product_path):
        self.sat_product_path = sat_product_path

    def create(self) -> SatProductType:
        if Sentinel1IWPathHandler().matches(self.sat_product_path):
            return SatProductType.SENTINEL_1_IW
        elif Sentinel1EWPathHandler().matches(self.sat_product_path):
            return SatProductType.SENTINEL_1_EW
        elif Sentinel2L1CPathHandler().matches(self.sat_product_path):
            return SatProductType.SENTINEL_2_L1C
        elif LandsatPathHandler().matches(self.sat_product_path):
            return SatProductType.LANDSAT
        else:
            raise NotImplementedError(self.sat_product_path)
