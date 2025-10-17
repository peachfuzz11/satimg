import os
import re
import typing
from pathlib import Path

from handlers.product_handler import ProductHandler
from products.base.product import Product
from products.iceye.iceye_scan_product import IceyeScanProduct
from products.iceye.iceye_stripmap_product import IceyeStripmapProduct


class IceyeHandler(ProductHandler):
    ICEYE_SM = re.compile(r'ICEYE.*_GRD_SM_.*\.tif$')
    ICEYE_SC = re.compile(r'ICEYE.*_GRD_SC_.*\.tif$')

    def create_product(self, product_path: typing.Union[Path, str]) -> Product:
        if os.path.isdir(product_path):
            if any([o for o in os.listdir(str(product_path)) if any(re.findall(self.ICEYE_SM, o))]):
                return IceyeStripmapProduct(product_path)
            if any([o for o in os.listdir(str(product_path)) if any(re.findall(self.ICEYE_SC, o))]):
                return IceyeScanProduct(product_path)
