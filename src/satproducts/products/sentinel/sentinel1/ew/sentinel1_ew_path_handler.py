import re

from satproducts.products.base.product_path_handler import ProductPathHandler


class Sentinel1EWPathHandler(ProductPathHandler):
    PATTERN = re.compile(r"S1[ABCD]_EW_GRDM_1SD[HV]_\d{8}T\d{6}_\d{8}T\d{6}.*\.SAFE$")
