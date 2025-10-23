import re

from satproducts.products.base.product_path_handler import ProductPathHandler


class Sentinel2L1CPathHandler(ProductPathHandler):
    PATTERN = re.compile(r"S2[ABCD]_MSIL1C_\d{8}T\d{6}_.*_\d{8}T\d{6}\.SAFE$")
