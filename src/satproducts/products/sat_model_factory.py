import typing

from satproducts.products.base.connector import Connector
from satproducts.products.landsat.landsat_l1gt_connector import LandsatL1GTConnector
from satproducts.products.sat_product_type import SatProductType
from satproducts.products.sentinel.sentinel1.ew.sentinel1_ew_connector import Sentinel1EWConnector
from satproducts.products.sentinel.sentinel1.iw.sentinel1_iw_connector import Sentinel1IWConnector
from satproducts.products.sentinel.sentinel2.sentinel2_connector import Sentinel2Connector


class SatModelFactory:

    def __init__(self, sat_product_type: SatProductType):
        self.sat_product_type = sat_product_type

    def create(self) -> typing.Type[Connector]:
        if SatProductType.SENTINEL_1_IW == self.sat_product_type:
            return Sentinel1IWConnector
        elif SatProductType.SENTINEL_1_EW == self.sat_product_type:
            return Sentinel1EWConnector
        elif SatProductType.SENTINEL_2_L1C == self.sat_product_type:
            return Sentinel2Connector
        elif SatProductType.LANDSAT == self.sat_product_type:
            return LandsatL1GTConnector
        else:
            raise NotImplementedError(self.sat_product_type)
