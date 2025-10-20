import os
import re
import typing
from pathlib import Path
from xml.etree import ElementTree

from products.base.product_handler import ProductHandler
from products.base.product import Product
from products.rcm.rcm_100m_product import RCM100mProduct
from products.rcm.rcm_50m_product import RCM50mProduct
from products.rcm.rcm_5m_product import RCM5mProduct
from products.rcm.rcm_low_noise_product import RCM50mLowNoiseProduct
from products.rcm.rcm_ship_detection_product import RCMShipDetectionProduct


class RCMHandler(ProductHandler):

    RCM_50M_LOW_NOISE = re.compile(r"RCM[\d].*_SC50M[A-Z]_.*(HH_HV|VV_VH)_GRD$")
    RCM_50M = re.compile(r"RCM[\d].*_SC50M[A-Z]_.*")
    RCM_100M = re.compile(r"RCM[\d].*_SC100M[A-Z]_.*")
    RCM_SHIP_DETECTION = re.compile(r"RCM[\d].*_SCSDA_.*")
    RCM_5M = re.compile(r"RCM[\d].*_5MCP2_.*")

    def create_product(self, product_path: typing.Union[Path, str]) -> Product:

        if not any(re.findall(r"RCM", str(product_path))):
            return None
        try:
            if self.is_rcm_50m_low_noise(product_path):
                return RCM50mLowNoiseProduct(product_path)

            if self.is_rcm_50m(product_path):
                return RCM50mProduct(product_path)

            if self.is_rcm_100m(product_path):
                return RCM100mProduct(product_path)

            if self.is_rcm_ship_detection(product_path):
                return RCMShipDetectionProduct(product_path)

            if self.is_rcm_5m(product_path):
                return RCM5mProduct(product_path)
        except:
            pass

    def is_rcm_50m_low_noise(self, product_path) -> bool:
        """Check if the product is a 50m Low Noise RCM product."""
        if any(re.findall(self.RCM_50M_LOW_NOISE, str(product_path))):
            return True
        elif os.path.isdir(product_path):
            for product_type, mode in self.extract_type_and_mode(product_path):
                if product_type == "GRD" and mode == "SC50M" and self.extract_beam_mode(product_path) == "Medium Resolution 50m":
                    return True
        return False

    def is_rcm_50m(self, product_path) -> bool:
        """Check if the product is a 50m RCM product."""
        if any(re.findall(self.RCM_50M, str(product_path))):
            return True
        elif os.path.isdir(product_path):
            for product_type, mode in self.extract_type_and_mode(product_path):
                if product_type == "GRD" and mode == "SC50M":
                    return True
        return False

    def is_rcm_100m(self, product_path) -> bool:
        """Check if the product is a 100m RCM product."""
        if any(re.findall(self.RCM_100M, str(product_path))):
            return True
        elif os.path.isdir(product_path):
            for product_type, mode in self.extract_type_and_mode(product_path):
                if product_type == "GRD" and mode == "SC100M":
                    return True
        return False

    def is_rcm_ship_detection(self, product_path) -> bool:
        """Check if the product is a Ship Detection RCM product."""
        if any(re.findall(self.RCM_SHIP_DETECTION, str(product_path))):
            return True
        elif os.path.isdir(product_path):
            for product_type, mode in self.extract_type_and_mode(product_path):
                if product_type == "GRD" and mode == "SCSDA":
                    return True
        return False

    def is_rcm_5m(self, product_path) -> bool:
        """Check if the product is a 5m RCM product."""
        if any(re.findall(self.RCM_5M, str(product_path))):
            return True
        elif os.path.isdir(product_path):
            for product_type, mode in self.extract_type_and_mode(product_path):
                if product_type == "SLC" and mode == "5MCP2":
                    return True
        return False

    def extract_type_and_mode(self, product_path):
        """Extract product type and mode from XML transformers."""
        metadata_file = os.path.join(product_path, "transformers", "product.xml")

        if not os.path.exists(metadata_file):
            return

        root = ElementTree.parse(metadata_file).getroot()

        product_type_elem = root.find(".//{rcmGsProductSchema}productType")
        beam_mode_elem = root.find(".//{rcmGsProductSchema}beamModeMnemonic")

        product_type = product_type_elem.text if product_type_elem is not None else None
        mode = beam_mode_elem.text if beam_mode_elem is not None else None

        if product_type and mode:
            yield product_type, mode

    def extract_beam_mode(self, product_path) -> typing.Optional[str]:
        """Extract beam mode from product transformers."""
        metadata_file = os.path.join(product_path, "transformers", "product.xml")

        if not os.path.exists(metadata_file):
            return None

        root = ElementTree.parse(metadata_file).getroot()
        beam_mode = root.find(".//{rcmGsProductSchema}beamMode")

        return beam_mode.text if beam_mode is not None else None
