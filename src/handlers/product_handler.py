import abc
import typing
from pathlib import Path

from products.base.product import Product


class ProductHandler(abc.ABC):
    def __init__(self, successor: "ProductHandler" = None):
        self.successor = successor

    def handle(self, product_path):
        # Attempt to resolve it using current handler
        product = self.create_product(product_path)
        if product is None:
            if self.successor is None:
                raise Exception(f"Could not resolve {product_path}")
            # Recursive call
            product = self.successor.handle(product_path)
        return product

    @abc.abstractmethod
    def create_product(self, product_path: typing.Union[Path, str]) -> Product:
        pass
