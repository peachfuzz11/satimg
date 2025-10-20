import os.path
import re
import typing
from pathlib import Path
from xml.etree import ElementTree

from products.base.product_handler import ProductHandler
from products.base.product import Product
from products.sentinel.sentinel1.sentinel1_ew_product import Sentinel1EWProduct
from products.sentinel.sentinel1.sentinel1_iw_product import Sentinel1IWProduct
from products.sentinel.sentinel2.sentinel2_l1c_product import Sentinel2L1CProduct


class SentinelHandler(ProductHandler):

    SENTINEL1_IW = re.compile(r"S1[ABCD]_IW_GRDH_1SD[HV]_\d{8}T\d{6}_\d{8}T\d{6}.*\.SAFE$")
    SENTINEL1_EW = re.compile(r"S1[ABCD]_EW_GRDM_1SD[HV]_\d{8}T\d{6}_\d{8}T\d{6}.*\.SAFE$")
    SENTINEL2_L1C = re.compile(r"S2[ABCD]_MSIL1C_\d{8}T\d{6}_.*_\d{8}T\d{6}\.SAFE$")

    def create_product(self, product_path: typing.Union[Path, str]) -> Product:
        if self.is_sentinel1_iw_hh_hv(product_path):
            return Sentinel1IWProduct(product_path)
        if self.is_sentinel1_iw_vv_vh(product_path):
            return Sentinel1IWProduct(product_path)
        if self.is_sentinel1_ew(product_path):
            return Sentinel1EWProduct(product_path)
        if self.is_sentinel2_l1c(product_path):
            return Sentinel2L1CProduct(product_path)
        return None

    def is_sentinel1_iw_hh_hv(self, product_path) -> bool:
        if any(re.findall(self.SENTINEL1_IW, str(product_path))):
            return True
        elif os.path.isdir(product_path):
            for product_type, mode in self.extract_type_and_mode(product_path):
                if product_type == "GRD" and mode == "IW":
                    return True
        else:
            return False

    def extract_type_and_mode(self, product_path):
        annotation_dir = os.path.join(product_path, "annotation")
        if os.path.exists(annotation_dir) and os.path.isdir(annotation_dir):
            annotations = [f for f in os.listdir(annotation_dir) if ".xml" in f]
            for annotation in annotations:
                root = ElementTree.parse(os.path.join(annotation_dir, annotation)).getroot()
                product_type = root.find(".//productType").text
                mode = root.find(".//mode").text
                yield product_type, mode

    def is_sentinel1_iw_vv_vh(self, product_path) -> bool:
        return self.is_sentinel1_iw_hh_hv(product_path)

    def is_sentinel1_ew(self, product_path):
        if any(re.findall(self.SENTINEL1_EW, str(product_path))):
            return True
        elif os.path.isdir(product_path):
            for product_type, mode in self.extract_type_and_mode(product_path):
                if product_type == "GRD" and mode == "EW":
                    return True
        else:
            return False

    def is_sentinel2_l1c(self, product_path):
        if any(re.findall(self.SENTINEL2_L1C, str(product_path))):
            return True
        elif os.path.isdir(product_path):
            mtd = os.path.join(product_path, "MTD_MSIL1C.xml")
            if os.path.exists(mtd):
                root = ElementTree.parse(mtd).getroot()
                product_type = root.find(".//PRODUCT_TYPE").text
                if product_type == "S2MSI1C":
                    return True
        else:
            return False
