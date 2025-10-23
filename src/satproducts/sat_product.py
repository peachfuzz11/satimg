import typing
from pathlib import Path

from satproducts.products.base.connector import Connector
from satproducts.products.base.product import Product
from satproducts.products.sat_connector_factory import SatConnectorFactory
from satproducts.products.sat_product_factory import SatProductFactory
from satproducts.products.sat_product_type import SatProductType
from satproducts.products.sat_product_type_factory import SatProductTypeFactory


class SatProduct:

    def __init__(self, product_path: typing.Union[Path, str]):
        self._product_path = product_path
        self._sat_product_type = _get_type(product_path)

    def get_type(self) -> SatProductType:
        return self._sat_product_type

    def get_product(self) -> Product:
        product = SatProductFactory(self._sat_product_type).create()
        return product(self._product_path)

    def get_connector(self) -> Connector:
        connector = SatConnectorFactory(self._sat_product_type).create()
        return connector()


def _get_type(sat_product_path) -> SatProductType:
    factory = SatProductTypeFactory(sat_product_path)
    return factory.create()
