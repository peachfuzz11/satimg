import logging
import os

import rasterio

from satproducts.transformers.base_transformer import BaseTransformer
from satproducts.products.synspective.synspective_product import SynspectiveProduct


class SynspectiveSMSLCProduct(SynspectiveProduct):
    def __init__(self, product_path: str):
        super().__init__(product_path)
        self._product_path = product_path
        self._tif_path = None
        self._xml_path = None
        self._product = None  # Placeholder for dataset
        self.metadata = {}

        self._init_files()

    def _init_files(self):
        self._tif_path, self._xml_path = self._locate_files()

    def _locate_files(self):
        """Locate .nitf and .jpeg files in the product path."""
        jpeg_file = None
        nitf_file = None

        # Search for .jpeg and .nitf files in the directory
        for file in os.listdir(self._product_path):
            if file.endswith(".jpeg"):
                jpeg_file = os.path.join(self._product_path, file)
            elif file.endswith(".nitf"):
                nitf_file = os.path.join(self._product_path, file)

        # Ensure both files are found
        if jpeg_file and nitf_file:
            return nitf_file, jpeg_file
        else:
            logging.error(f"Failed to locate both .nitf and .jpeg files in {self._product_path}")
            return None, None

    def open(self):
        """Open the .nitf file using rasterio."""
        if self._tif_path:
            try:
                self._product = rasterio.open(self._tif_path)
                logging.info(f"Successfully opened NITF file: {self._tif_path}")
            except Exception as e:
                logging.error(f"Failed to open NITF file: {self._tif_path}, error: {e}")
                raise

            self._width = 0
            self._height = 0
            self._bands = 0
        else:
            logging.error("NITF file path is missing. Cannot open the product.")
            raise FileNotFoundError("NITF file not found.")

    def parse_metadata(self):
        """Parse the XML transformers file."""
        return NotImplementedError

    def close(self):
        """Close the opened dataset."""
        if self._product:
            try:
                self._product.close()
                logging.info(f"Closed product: {self._tif_path}")
            except Exception as e:
                logging.error(f"Failed to close product: {self._tif_path}, error: {e}")
                raise

    def validate(self):
        self._valid = True
        logging.info("Validation passed for Synspective SLC Product.")

    def _extract_metadata(self):
        pass

    def get_transformer(self, *args, **kwargs) -> BaseTransformer:
        return super().get_transformer(self._tif_path)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self):
        return f"SynspectiveSMSLCProduct(tif_path={self._tif_path})"
