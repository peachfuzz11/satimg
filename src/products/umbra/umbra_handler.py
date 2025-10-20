import os
import re
import typing
from pathlib import Path
from xml.etree import ElementTree

from products.base.product_handler import ProductHandler
from products.base.product import Product
from products.umbra.umbra_product import UmbraProduct


class UmbraHandler(ProductHandler):
    """
    Handler for Umbra satellite data products.

    All available UMBRA data is delivered in processed folders, i.e., each folder has level 0, 1, 2, etc.
    """
    UMBRA_PATTERN = re.compile(r"Umbra.*")

    def create_product(self, product_path: typing.Union[Path, str]) -> typing.Optional[Product]:
        if not any(re.findall(self.UMBRA_PATTERN, str(product_path))):
            return None
        if self.is_umbra_product(product_path):
            return UmbraProduct(product_path)

    def is_umbra_product(self, product_path) -> bool:
        """Check if the product is an Umbra product."""
        if any(re.findall(self.UMBRA_PATTERN, str(product_path))):
            return True
        elif os.path.isdir(product_path):
            metadata_file = os.path.join(product_path, "transformers", "product.xml")
            if os.path.exists(metadata_file):
                root = ElementTree.parse(metadata_file).getroot()
                product_name_elem = root.find(".//{umbraProductSchema}productName")
                if product_name_elem is not None and "Umbra" in product_name_elem.text:
                    return True
        return False
