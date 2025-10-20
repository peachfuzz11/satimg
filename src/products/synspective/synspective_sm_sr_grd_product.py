import os
import xml.etree.ElementTree as ET

import rioxarray

from transformers.base_transformer import BaseTransformer
from products.synspective.synspective_product import SynspectiveProduct


class SynspectiveSMSRGRDProduct(SynspectiveProduct):
    """ """

    def __init__(self, product_path: str):
        # Ensure that the product_path is passed to the parent class
        super().__init__(product_path)
        self._product_path = product_path
        self._product = None  # rioxarray dataset
        self.metadata = {}

        self._tif_path, self._xml_path = self._locate_files()

    def open(self):
        """Open the GeoTIFF file using rioxarray."""
        self._product = rioxarray.open_rasterio(self._tif_path, chunks=True)
        self._extract_metadata()

    def get_transformer(self, **kwargs) -> BaseTransformer:
        return super().get_transformer(self._tif_path)

    def _locate_files(self):
        """Automatically locate the .tif and .xml files based on the product path."""
        files = os.listdir(self._product_path)
        tif_file = None
        xml_file = None

        for file in files:
            if file.endswith(".tif"):
                tif_file = os.path.join(self._product_path, file)
            elif file.endswith(".xml"):
                xml_file = os.path.join(self._product_path, file)

        if not tif_file or not xml_file:
            raise FileNotFoundError(f"Required .tif or .xml file not found in {self._product_path}")

        return tif_file, xml_file

    def parse_metadata(self):
        """Parse the XML transformers file."""

        tree = ET.parse(self._xml_path)
        root = tree.getroot()

        def get_text_or_none(element, path, namespaces):
            """Helper function to get text from an element, or None if the element doesn't exist."""
            result = element.find(path, namespaces=namespaces)
            return result.text if result is not None else None

        # Extract useful transformers fields from the XML file
        meta_data = root.find(".//eop:EarthObservationMetaData", namespaces={"eop": "http://earth.esa.int/eop"})
        processing_info = meta_data.find(".//eop:ProcessingInformation", namespaces={"eop": "http://earth.esa.int/eop"})
        acquisition_info = root.find(".//sar:Acquisition", namespaces={"sar": "http://earth.esa.int/sar"})
        platform_info = root.find(".//eop:EarthObservationEquipment/eop:platform",
                                  namespaces={"eop": "http://earth.esa.int/eop"})

        # Extract transformers fields using the helper function
        self.metadata = {
            "creation_date": get_text_or_none(meta_data, "eop:creationDate",
                                              namespaces={"eop": "http://earth.esa.int/eop"}),
            "processing_date": get_text_or_none(processing_info, "eop:processingDate",
                                                namespaces={"eop": "http://earth.esa.int/eop"}),
            "processing_level": get_text_or_none(processing_info, "eop:processingLevel",
                                                 namespaces={"eop": "http://earth.esa.int/eop"}),
            "range_pixel_spacing": get_text_or_none(
                processing_info, "sar:sarProcessingParameter/sar:rangePixelSpacing",
                namespaces={"sar": "http://earth.esa.int/sar"}
            ),
            "azimuth_pixel_spacing": get_text_or_none(
                processing_info, "sar:sarProcessingParameter/sar:azimuthPixelSpacing",
                namespaces={"sar": "http://earth.esa.int/sar"}
            ),
            "off_nadir_angle": get_text_or_none(
                meta_data,
                ".//eop:SpecificInformation[eop:localAttribute='offnadirAngle']/eop:localValue",
                namespaces={"eop": "http://earth.esa.int/eop"},
            ),
            "calibration_factor": get_text_or_none(
                meta_data,
                ".//eop:SpecificInformation[eop:localAttribute='calibrationFactor']/eop:localValue",
                namespaces={"eop": "http://earth.esa.int/eop"},
            ),
            "nesz_max_power": get_text_or_none(
                meta_data,
                ".//eop:SpecificInformation[eop:localAttribute='neszMaximumPower']/eop:localValue",
                namespaces={"eop": "http://earth.esa.int/eop"},
            ),
            "nesz_min_power": get_text_or_none(
                meta_data,
                ".//eop:SpecificInformation[eop:localAttribute='neszMinimumPower']/eop:localValue",
                namespaces={"eop": "http://earth.esa.int/eop"},
            ),
            "acquisition_prf": get_text_or_none(acquisition_info, "sar:acquisitionPRF",
                                                namespaces={"sar": "http://earth.esa.int/sar"}),
            "carrier_frequency": get_text_or_none(acquisition_info, "sar:carrierFrequency",
                                                  namespaces={"sar": "http://earth.esa.int/sar"}),
            "incidence_angle_min": get_text_or_none(acquisition_info, "sar:minimumIncidenceAngle",
                                                    namespaces={"sar": "http://earth.esa.int/sar"}),
            "incidence_angle_max": get_text_or_none(acquisition_info, "sar:maximumIncidenceAngle",
                                                    namespaces={"sar": "http://earth.esa.int/sar"}),
            "satellite_name": get_text_or_none(platform_info, "eop:shortName",
                                               namespaces={"eop": "http://earth.esa.int/eop"}),
            "satellite_serial": get_text_or_none(platform_info, "eop:serialIdentifier",
                                                 namespaces={"eop": "http://earth.esa.int/eop"}),
            "orbit_type": get_text_or_none(platform_info, "eop:orbitType",
                                           namespaces={"eop": "http://earth.esa.int/eop"}),
            "orbit_direction": get_text_or_none(acquisition_info, "eop:orbitDirection",
                                                namespaces={"eop": "http://earth.esa.int/eop"}),
        }

        # Handle footprint geometry and convert it to GeoJSON-like format
        footprint = root.find(".//gml:LinearRing/gml:posList", namespaces={"gml": "http://www.opengis.net/gml"})
        if footprint is not None:
            coordinates = footprint.text.strip().split()
            coordinates = [(float(coordinates[i + 1]), float(coordinates[i])) for i in range(0, len(coordinates), 2)]
            self.metadata["geometry"] = {"type": "Polygon", "coordinates": [coordinates]}
        else:
            self.metadata["geometry"] = None

    def close(self):
        """Close the opened dataset."""
        self._product.close()
        self._product = None

    def validate(self):
        pass

    def _extract_metadata(self):
        self.parse_metadata()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self):
        return f"SynspectiveSMSRGRDProduct(tif_path={self._tif_path})"
