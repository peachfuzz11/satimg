import typing
from pathlib import Path

from products.base.product import Product
from products.base.product_handler import ProductHandler
from products.landsat.landsat_product import LandsatProduct


class LandsatHandler(ProductHandler):
    def create_product(self, product_path: typing.Union[Path, str]) -> Product:
        try:
            return LandsatProduct(product_path)
        except Exception as e:
            return None
