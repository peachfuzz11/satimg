import os
import tempfile
import typing
import zipfile
from contextlib import contextmanager
from pathlib import Path

from satproducts.products.base.product import Product
from satproducts.products.sat_product_factory import SatProductFactory
from satproducts.products.sat_product_type import SatProductType
from satproducts.products.sat_product_type_factory import SatProductTypeFactory


class SatProduct:

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

    @staticmethod
    @contextmanager
    def from_zip(zip_path: str, dest=None):
        if dest is None:
            raise ValueError("destination for tmp dir not specified")
        with tempfile.TemporaryDirectory(dir=dest) as temp_dir:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)
            product_folder_path: str = next(os.path.join(temp_dir, o) for o in os.listdir(temp_dir))
            yield SatProduct.from_path(product_folder_path)
