import xml.etree.ElementTree as ET
from typing import Optional, Dict, Any, List
import warnings
import os


def check_image_order(image_list: List[str]) -> List[str]:
    """
    Determine the order of polarisation bands in a list of image files.

    Args:
        image_list (List[str]): List of image file names.

    Returns:
        List[str]: Ordered list of detected polarisation bands (e.g., ['VV', 'VH']).
    """
    order = []
    for image in image_list:
        image = image.split("/")[-1]
        if "vv" in image.lower():
            order.append("VV")
        elif "vh" in image.lower():
            order.append("VH")
        elif "hh" in image.lower():
            order.append("HH")
        elif "hv" in image.lower():
            order.append("HV")
    return order


class RCMMetadataLoader:
    """
    Class for loading and parsing RCM metadata from product.xml.

    Attributes:
        product_path (str): Path to the RCM product folder.
    """

    def __init__(self, product_path: Optional[str] = None):
        self._product_path = product_path
        self._ns = {"rcm": "rcmGsProductSchema"}  # XML namespace for RCM files

        self._locate_files()

    @property
    def product_path(self) -> Optional[str]:
        """Getter for product path."""
        return self._product_path

    @product_path.setter
    def product_path(self, path: str):
        """Setter for product path."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"The specified path does not exist: {path}")
        self._product_path = path

    def load_product_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Load metadata from the RCM product.xml file.

        Returns:
            dict: Parsed metadata as a structured dictionary.
        """
        if not self._product_path:
            raise ValueError("Product path is not set.")

        product_file = os.path.join(self._product_path, "metadata", "product.xml")
        if not os.path.exists(product_file):
            raise FileNotFoundError(f"product.xml not found in {self._product_path}")

        try:
            tree = ET.parse(product_file)
            root = tree.getroot()

            self.metadata = {
                "file_path": self._product_path,
                "ProductType": root.find("rcm:sourceAttributes/rcm:beamMode", self._ns).text,
                "Satellite": root.find("rcm:sourceAttributes/rcm:satellite", self._ns).text,
                "Start": root.findtext("rcm:sourceAttributes/rcm:rawDataStartTime", namespaces=self._ns),
                "Shape": None,  # self._product.rio.shape,  #TODO add in the child classes
                "resolution": None,
                "Mode": root.find("rcm:sourceAttributes/rcm:radarParameters/rcm:acquisitionType", self._ns).text,
            }

            self.advanced_metadata = {
                "productId": root.find("rcm:productId", self._ns).text,
                "productApplication": root.find("rcm:productApplication", self._ns).text,
                "documentIdentifier": root.find("rcm:documentIdentifier", self._ns).text,
                "securityClassification": root.find("rcm:securityAttributes/rcm:securityClassification", self._ns).text,
                "specialHandlingRequired": root.find("rcm:securityAttributes/rcm:specialHandlingRequired", self._ns).text,
                "sensor": root.find("rcm:sourceAttributes/rcm:sensor", self._ns).text,
                "polarizationDataMode": root.find("rcm:sourceAttributes/rcm:polarizationDataMode", self._ns).text,
                "beamModeMnemonic": root.find("rcm:sourceAttributes/rcm:beamModeMnemonic", self._ns).text,
                "rawDataStartTime": root.find("rcm:sourceAttributes/rcm:rawDataStartTime", self._ns).text,
                "polarizations": root.find("rcm:sourceAttributes/rcm:radarParameters/rcm:polarizations", self._ns).text,
                "beams": root.find("rcm:sourceAttributes/rcm:radarParameters/rcm:beams", self._ns).text,
                "product_info": {
                    "productId": root.findtext("rcm:productId", namespaces=self._ns),
                    "productAnnotation": root.findtext("rcm:productAnnotation", namespaces=self._ns),
                    "productApplication": root.findtext("rcm:productApplication", namespaces=self._ns),
                    "documentIdentifier": root.findtext("rcm:documentIdentifier", namespaces=self._ns),
                },
                "security_attributes": {
                    "securityClassification": root.findtext("rcm:securityAttributes/rcm:securityClassification", namespaces=self._ns),
                    "specialHandlingRequired": root.findtext("rcm:securityAttributes/rcm:specialHandlingRequired", namespaces=self._ns),
                    "specialHandlingInstructions": root.findtext("rcm:securityAttributes/rcm:specialHandlingInstructions", namespaces=self._ns),
                },
                "source_attributes": {
                    "satellite": root.findtext("rcm:sourceAttributes/rcm:satellite", namespaces=self._ns),
                    "sensor": root.findtext("rcm:sourceAttributes/rcm:sensor", namespaces=self._ns),
                    "polarizationDataMode": root.findtext("rcm:sourceAttributes/rcm:polarizationDataMode", namespaces=self._ns),
                    "downlinkSegmentId": root.findtext("rcm:sourceAttributes/rcm:downlinkSegmentId", namespaces=self._ns),
                    "inputDatasetFacilityId": root.findtext("rcm:sourceAttributes/rcm:inputDatasetFacilityId", namespaces=self._ns),
                    "beamMode": root.findtext("rcm:sourceAttributes/rcm:beamMode", namespaces=self._ns),
                    "beamModeDefinitionId": root.findtext("rcm:sourceAttributes/rcm:beamModeDefinitionId", namespaces=self._ns),
                    "beamModeVersion": root.findtext("rcm:sourceAttributes/rcm:beamModeVersion", namespaces=self._ns),
                    "beamModeMnemonic": root.findtext("rcm:sourceAttributes/rcm:beamModeMnemonic", namespaces=self._ns),
                },
                "radar_parameters": {
                    "acquisitionType": root.findtext("rcm:sourceAttributes/rcm:radarParameters/rcm:acquisitionType", namespaces=self._ns),
                    "beams": root.findtext("rcm:sourceAttributes/rcm:radarParameters/rcm:beams", namespaces=self._ns).split(),
                    "polarizations": root.findtext("rcm:sourceAttributes/rcm:radarParameters/rcm:polarizations", namespaces=self._ns).split(),
                    "pulses": list(map(int, root.findtext("rcm:sourceAttributes/rcm:radarParameters/rcm:pulses", namespaces=self._ns).split())),
                    "radarCenterFrequency": float(
                        root.findtext("rcm:sourceAttributes/rcm:radarParameters/rcm:radarCenterFrequency", namespaces=self._ns)
                    ),
                },
            }

            # Parse PRF information
            prf_elements = root.findall("rcm:sourceAttributes/rcm:radarParameters/rcm:prfInformation", self._ns)
            prf_info = []
            for prf in prf_elements:
                prf_info.append(
                    {
                        "beam": prf.get("beam"),
                        "pole": prf.get("pole"),
                        "burst": int(prf.get("burst")),
                        "rawLine": int(prf.findtext("rcm:rawLine", namespaces=self._ns)),
                        "pulseRepetitionFrequency": float(prf.findtext("rcm:pulseRepetitionFrequency", namespaces=self._ns)),
                    }
                )
            self.advanced_metadata["radar_parameters"]["prf_info"] = prf_info
            # self._locate_files()
            #  self.tiff_paths = self.files_dict["tif"]
            order = check_image_order(self.files_dict["tif"])
            self.metadata["band_order"] = order

            return None

        except Exception as e:
            warnings.warn(f"Error loading product metadata: {e}")
            return None

    def load_calibration_incidence_angles(self) -> Optional[Dict[str, Any]]:
        """
        Load incidence angles from RCM calibration/incidenceAngles.xml.

        Returns:
            dict: Parsed incidence angle metadata as a structured dictionary.
        """
        if not self._product_path:
            raise ValueError("Product path is not set.")

        incidence_file = os.path.join(self._product_path, "metadata", "calibration", "incidenceAngles.xml")
        if not os.path.exists(incidence_file):
            raise FileNotFoundError(f"incidenceAngles.xml not found in {self._product_path}/calibration")

        try:
            tree = ET.parse(incidence_file)
            root = tree.getroot()

            # Extract data from XML
            pixel_first_angle_value = int(root.find("rcm:pixelFirstAnglesValue", self._ns).text)
            step_size = int(root.find("rcm:stepSize", self._ns).text)
            number_of_values = int(root.find("rcm:numberOfValues", self._ns).text)
            angles = [float(angle.text) for angle in root.findall("rcm:angles", self._ns)]

            # Construct result
            incidence_angles_data = {
                "pixel_first_angle_value": pixel_first_angle_value,
                "step_size": step_size,
                "number_of_values": number_of_values,
                "angles": angles,
                "units": "deg",
            }

            self.calibration_metadata = {"incidence_angles": incidence_angles_data}
            return None

        except Exception as e:
            warnings.warn(f"Error loading incidence angles: {e}")
            return None

    def load_lut(self, lut_file: str, lut_name: str) -> Optional[Dict[str, Any]]:
        """
        Load lookup table (LUT) data from a specified LUT XML file and add it to calibration metadata.

        Args:
            lut_file (str): Path to the LUT XML file.
            lut_name (str): Name to store the LUT in calibration metadata.

        Returns:
            dicpasst: Parsed LUT metadata including pixel first value, step size, offset, and gains.
        """
        if not os.path.exists(lut_file):
            raise FileNotFoundError(f"LUT file not found: {lut_file}")

        try:
            tree = ET.parse(lut_file)
            root = tree.getroot()

            # Extract LUT metadata
            pixel_first_value = int(root.find("rcm:pixelFirstLutValue", self._ns).text)
            step_size = int(root.find("rcm:stepSize", self._ns).text)
            number_of_values = int(root.find("rcm:numberOfValues", self._ns).text)
            offset = int(root.find("rcm:offset", self._ns).text)
            gains = [float(value) for value in root.find("rcm:gains", self._ns).text.split()]

            lut_data = {
                "pixel_first_value": pixel_first_value,
                "step_size": step_size,
                "number_of_values": number_of_values,
                "offset": offset,
                "gains": gains,
            }

            # Add LUT data to calibration metadata
            if not hasattr(self, "calibration_metadata"):
                self.calibration_metadata = {}

            self.calibration_metadata[lut_name] = lut_data

            return None

        except Exception as e:
            warnings.warn(f"Error loading LUT data from {lut_file}: {e}")
            return None

    def load_all_luts(self):
        """
        Load all LUT files (Beta, Gamma, Sigma) for HH and HV polarizations
        and add them to calibration metadata.
        """

        try:

            lut_files = {
                "lutBeta_HH": os.path.join(self._product_path, "metadata", "calibration", "lutBeta_HH.xml"),
                "lutBeta_HV": os.path.join(self._product_path, "metadata", "calibration", "lutBeta_HV.xml"),
                "lutGamma_HH": os.path.join(self._product_path, "metadata", "calibration", "lutGamma_HH.xml"),
                "lutGamma_HV": os.path.join(self._product_path, "metadata", "calibration", "lutGamma_HV.xml"),
                "lutSigma_HH": os.path.join(self._product_path, "metadata", "calibration", "lutSigma_HH.xml"),
                "lutSigma_HV": os.path.join(self._product_path, "metadata", "calibration", "lutSigma_HV.xml"),
            }
            for lut_name, lut_path in lut_files.items():
                self.load_lut(lut_path, lut_name)
        except Exception as e:
            pass

    def _locate_files(self):
        """Automatically locate the .tif, .xml, and other important files."""
        files_dict = {
            "tif": [],  # List to store multiple .tif image files
            "xml": [],  # List to store multiple product .xml files
            "safe": None,  # SAFE metadata (if present)
            "noise_levels_hh": [],  # List for HH noise level files
            "noise_levels_hv": [],  # List for HV noise level files
            "noise_levels_vv": [],  # List for VV noise level files
            "noise_levels_vh": [],  # List for VH noise level files
            "calibration_hh": [],  # List for HH calibration files
            "calibration_hv": [],  # List for HV calibration files
            "calibration_vv": [],  # List for VV calibration files
            "calibration_vh": [],  # List for VH calibration files
            "annotation_hh": [],  # List for HH annotation files
            "annotation_hv": [],  # List for HV annotation files
            "annotation_vv": [],  # List for VV annotation files
            "annotation_vh": [],  # List for VH annotation files
            "calibration_files": [],  # All files in the calibration folder
            "support_files": [],  # All files in the support folder
            "other_files": [],  # Capture any other relevant files
        }

        for root, dirs, files in os.walk(self._product_path):
            for file in files:
                file_path = os.path.join(root, file)

                # Identify all .tif files in the "imagery" folder
                if file.endswith(".tif") and "imagery" in root and not file.startswith("."):

                    files_dict["tif"].append(file_path)

                # Identify all product .xml files in the "metadata" folder
                elif file.endswith(".xml") and "metadata" in root:
                    if "product" in file.lower():
                        files_dict["xml"].append(file_path)

                    # Identify noise level files (HH, HV, VV, VH)
                    if "noiseLevels_HH" in file:
                        files_dict["noise_levels_hh"].append(file_path)
                    elif "noiseLevels_HV" in file:
                        files_dict["noise_levels_hv"].append(file_path)
                    elif "noiseLevels_VV" in file:
                        files_dict["noise_levels_vv"].append(file_path)
                    elif "noiseLevels_VH" in file:
                        files_dict["noise_levels_vh"].append(file_path)

                    # Identify calibration files (HH, HV, VV, VH)
                    if "calibration_HH" in file:
                        files_dict["calibration_hh"].append(file_path)
                    elif "calibration_HV" in file:
                        files_dict["calibration_hv"].append(file_path)
                    elif "calibration_VV" in file:
                        files_dict["calibration_vv"].append(file_path)
                    elif "calibration_VH" in file:
                        files_dict["calibration_vh"].append(file_path)

                    # Identify annotation files (HH, HV, VV, VH)
                    if "annotation_HH" in file:
                        files_dict["annotation_hh"].append(file_path)
                    elif "annotation_HV" in file:
                        files_dict["annotation_hv"].append(file_path)
                    elif "annotation_VV" in file:
                        files_dict["annotation_vv"].append(file_path)
                    elif "annotation_VH" in file:
                        files_dict["annotation_vh"].append(file_path)

                # Identify files in the "calibration" folder
                elif "calibration" in root:
                    files_dict["calibration_files"].append(file_path)

                # Identify files in the "support" folder
                elif "support" in root:
                    files_dict["support_files"].append(file_path)

                # Identify the .SAFE metadata file (if it exists)
                elif file.endswith(".SAFE"):
                    files_dict["safe"] = file_path

                # Capture any other relevant files
                else:
                    files_dict["other_files"].append(file_path)
        files_dict["tif"] = sorted(files_dict["tif"])
        # Return the dictionary with lists of files (or None where applicable)
        self.files_dict = files_dict
        del files_dict
        return None
