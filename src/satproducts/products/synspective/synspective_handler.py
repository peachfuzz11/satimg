import os
import re
import typing
from pathlib import Path
from xml.etree import ElementTree

from satproducts.products.base.product_handler import ProductHandler
from satproducts.products.base.product import Product
from satproducts.products.synspective.synspective_sm_grd_product import SynspectiveSMGRDProduct
from satproducts.products.synspective.synspective_sm_sr_grd_product import SynspectiveSMSRGRDProduct


class SynspectiveHandler(ProductHandler):
    
    SYN_GRD = re.compile(r"STRIX.*_SM_GRD")
    SYN_SR_GRD = re.compile(r"STRIX.*_SM_SR-GRD")

    def create_product(self, product_path: typing.Union[Path, str]) -> Product:
        
        if not any(re.findall(r"STRIX", str(product_path))):
            return None  # Skip this handler

        if self.is_synspective_sm_grd(product_path):
            return SynspectiveSMGRDProduct(product_path)

        if self.is_synspective_sm_sr_grd(product_path):
            return SynspectiveSMSRGRDProduct(product_path)
        
    def is_synspective_sm_grd(self, product_path) -> bool:
        """Check if the product is Synspective SM GRD."""
        if any(re.findall(self.SYN_GRD, str(product_path))):
            return True
        elif os.path.isdir(product_path):
            for processing_level, _ in self.extract_type_and_mode(product_path):
                if processing_level == "GRD":
                    return True
        return False

    def is_synspective_sm_sr_grd(self, product_path) -> bool:
        """Check if the product is Synspective SM SR-GRD."""
        if any(re.findall(self.SYN_SR_GRD, str(product_path))):
            return True
        elif os.path.isdir(product_path):
            for processing_level, _ in self.extract_type_and_mode(product_path):
                if processing_level == "GRD":
                    return True
        return False

    def is_synspective_sm_slc(self, product_path) -> bool:
        """Check if the product is Synspective SM SLC."""
        if any(re.findall(self.SYN_SLC, str(product_path))):
            return True
        elif os.path.isdir(product_path):
            for product_type, _ in self.extract_type_and_mode(product_path):
                if product_type == "SLC":
                    return True
        return False

    def extract_type_and_mode(self, product_path):
        """Extract product type and mode from the XML transformers."""

        namespaces = {
            'sar': 'http://earth.esa.int/sar',
            'eop': 'http://earth.esa.int/eop',
            'gml': 'http://www.opengis.net/gml',
            'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
        }

        # Find XML file in the product path
        files = os.listdir(product_path)
        for file in files:
            if file.endswith(".xml"):
                xml_file = os.path.join(product_path, file)

        # Parse XML file
        root = ElementTree.parse(xml_file).getroot()

        processing_level_elem = root.find(".//eop:processingLevel", namespaces)
        acquisition_subtype_elem = root.find(".//eop:acquisitionSubType", namespaces)

        # Handle cases where elements are not found
        processing_level = processing_level_elem.text if processing_level_elem is not None else None
        acquisition_subtype = acquisition_subtype_elem.text if acquisition_subtype_elem is not None else None

        if processing_level and acquisition_subtype:
            yield processing_level, acquisition_subtype
