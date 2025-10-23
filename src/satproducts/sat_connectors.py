import typing
from pathlib import Path

from satproducts.products.base.connector import Connector
from satproducts.products.sat_connector_factory import SatConnectorFactory
from satproducts.products.sat_product_type import SatProductType
from satproducts.products.sat_product_type_factory import SatProductTypeFactory


class SatConnectors:

    def __init__(self, sat_product_type: SatProductType):
        self._sat_product_type = sat_product_type

    @classmethod
    def from_path(cls, product_path: typing.Union[Path, str]):
        factory = SatProductTypeFactory(product_path)
        return cls(factory.create())

    def get_connector(self) -> Connector:
        connector = SatConnectorFactory(self._sat_product_type).create()
        return connector()
