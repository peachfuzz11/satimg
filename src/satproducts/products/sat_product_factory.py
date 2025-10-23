import typing

from satproducts.products.base.product import Product
from satproducts.products.landsat.landsat_product import LandsatProduct
from satproducts.products.sat_product_type import SatProductType
from satproducts.products.sentinel.sentinel1.ew.sentinel1_ew_product import Sentinel1EWProduct
from satproducts.products.sentinel.sentinel1.iw.sentinel1_iw_product import Sentinel1IWProduct
from satproducts.products.sentinel.sentinel2.sentinel2_l1c_product import Sentinel2L1CProduct


class SatProductFactory:

    def __init__(self, sat_product_type: SatProductType):
        self.sat_product_type = sat_product_type

    def create(self) -> typing.Type[Product]:
        if SatProductType.SENTINEL_1_IW == self.sat_product_type:
            return Sentinel1IWProduct
        elif SatProductType.SENTINEL_1_EW == self.sat_product_type:
            return Sentinel1EWProduct
        elif SatProductType.SENTINEL_2_L1C == self.sat_product_type:
            return Sentinel2L1CProduct
        elif SatProductType.LANDSAT == self.sat_product_type:
            return LandsatProduct
        else:
            raise NotImplementedError(self.sat_product_type)
