import typing

from satproducts.products.base.product import Product
from satproducts.products.landsat.landsat_product import LandsatProduct
from satproducts.products.sat_product_type import LANDSAT, SENTINEL_1_EW, SENTINEL_1_IW, SENTINEL_2_L1C
from satproducts.products.sentinel.sentinel1.ew.sentinel1_ew_product import Sentinel1EWProduct
from satproducts.products.sentinel.sentinel1.iw.sentinel1_iw_product import Sentinel1IWProduct
from satproducts.products.sentinel.sentinel2.sentinel2_l1c_product import Sentinel2L1CProduct


PRODUCT_MAP = {
    SENTINEL_1_IW: Sentinel1IWProduct,
    SENTINEL_1_EW: Sentinel1EWProduct,
    SENTINEL_2_L1C: Sentinel2L1CProduct,
    LANDSAT: LandsatProduct,
}


def get_product_class(product_type: str) -> typing.Type[Product]:
    product = PRODUCT_MAP.get(product_type)
    if product is None:
        raise NotImplementedError(product_type)
    return product