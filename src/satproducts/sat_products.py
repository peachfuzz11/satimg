import typing
from pathlib import Path

from satproducts.products.base.product import Product
from satproducts.products.sat_product_factory import SatProductFactory
from satproducts.products.sat_product_type import SatProductType
from satproducts.products.sat_product_type_factory import SatProductTypeFactory


class SatProducts:

    def __init__(self, product_path: typing.Union[Path, str], sat_product_type: SatProductType):
        self._product_path = product_path
        self._sat_product_type = sat_product_type

    @classmethod
    def from_path(cls, product_path: str):
        factory = SatProductTypeFactory(product_path)
        return cls(product_path, factory.create())

    def get_product(self) -> Product:
        product = SatProductFactory(self._sat_product_type).create()
        return product(self._product_path)
