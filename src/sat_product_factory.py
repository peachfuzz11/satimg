import typing
from pathlib import Path

from products.base.product import Product
from products.landsat.landsat_handler import LandsatHandler
from products.sentinel.sentinel_handler import SentinelHandler


class SatProductFactory:
    def __init__(self, product_path: typing.Union[Path, str]):
        self._product_path = product_path
        self._handler_chain = SentinelHandler(LandsatHandler())

    def create(self) -> Product:
        product = self._handler_chain.handle(self._product_path)
        return product
