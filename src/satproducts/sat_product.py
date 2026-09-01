import os
import tempfile
import typing
import zipfile
from contextlib import contextmanager
from pathlib import Path

from satproducts.products.base.product import Product
from satproducts.products.sat_product_factory import get_product_class
from satproducts.products.sat_product_type_factory import resolve_product_type


def create_from_path(product_path: str) -> Product:
    product_type = resolve_product_type(product_path)
    product_class = get_product_class(product_type)
    return product_class(product_path)


@contextmanager
def from_zip(zip_path: str, dest=None):
    if dest is None:
        raise ValueError("destination for tmp dir not specified")
    with tempfile.TemporaryDirectory(dir=dest) as temp_dir:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)
        product_folder_path: str = next(os.path.join(temp_dir, o) for o in os.listdir(temp_dir))
        yield create_from_path(product_folder_path)