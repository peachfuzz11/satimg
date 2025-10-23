import re

from satproducts.products.base.product_path_handler import ProductPathHandler


class LandsatPathHandler(ProductPathHandler):
    PATTERN = re.compile(r"LC(0[1-9])_L1(TP|GT)_\d+_\d+_\d+_\d+_T1$")
