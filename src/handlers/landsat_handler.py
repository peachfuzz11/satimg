import typing
from pathlib import Path

from handlers.product_handler import ProductHandler
from products.base.product import Product


class LandsatHandler(ProductHandler):
    def create_product(self, product_path: typing.Union[Path, str]) -> Product:

        pass
