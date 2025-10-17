import datetime
import os
import warnings
import xml.etree.ElementTree as ET
from typing import Optional, Dict, List, Any

import numpy as np


def get_files(directory, postfix: str = ".xml", include: Optional[str] = None) -> List[str]:
    """Retrieve files with a specific postfix, optionally filtered by a substring."""
    try:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"The directory {directory} does not exist.")
        files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(postfix)]
        if include:
            files = [f for f in files if include in f]
        if not files:
            return [None]
        return sorted(files)  # sorted since files are loaded sorted, but listdir returns an arbitraty order of files.
    except Exception as e:
        return [None]


def extract_text(element: ET.Element, tag: str, ns: Dict[str, str]) -> Optional[str]:
    """Extract text content from an XML element."""
    found = element.find(tag, ns)
    return found.text if found is not None else None


def extract_datetime(element: ET.Element, tag: str, ns: Dict[str, str]) -> Optional[datetime.datetime]:
    """Extract and parse datetime from an XML element."""
    text = extract_text(element, tag, ns)
    return datetime.datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%f") if text else None


def extract_float(element: ET.Element, tag: str, ns: Dict[str, str]) -> Optional[float]:
    """Extract and convert text to float from an XML element."""
    text = extract_text(element, tag, ns)
    return float(text) if text else None


def extract_int(element: ET.Element, tag: str, ns: Dict[str, str]) -> Optional[int]:
    """Extract and convert text to integer from an XML element."""
    text = extract_text(element, tag, ns)
    return int(text) if text else None


class Sentinel1MetadataLoader:
    """
    Class for loading and parsing Sentinel-1 metadata, calibration, and annotation files.

    Attributes:
        product_path (str): Path to the Sentinel-1 product folder.
    """

    def __init__(self, product_path: Optional[str] = None):
        self._product_path = product_path
        self._ns = {
            "xfdu": "urn:ccsds:schema:xfdu:1",
            "gml": "http://www.opengis.net/gml",
            "safe": "http://www.esa.int/safe/sentinel-1.0",
            "s1": "http://www.esa.int/safe/sentinel-1.0/sentinel-1",
            "s1sar": "http://www.esa.int/safe/sentinel-1.0/sentinel-1/sar",
            "s1sarl1": "http://www.esa.int/safe/sentinel-1.0/sentinel-1/sar/level-1",
            "s1path_safesarl2": "http://www.esa.int/safe/sentinel-1.0/sentinel-1/sar/level-2",
            "gx": "http://www.google.com/kml/ext/2.2",
        }

        self.product_path = product_path
        self.annotations_path = get_files(os.path.join(self._product_path, "annotation"))
        self.calibration_path = get_files(
            os.path.join(self._product_path, "annotation", "calibration"),
            include="calibration",
        )
        self.noise_path = get_files(
            os.path.join(self._product_path, "annotation", "calibration"),
            include="noise",
        )
        self.rfi_path = get_files(
            os.path.join(self._product_path, "annotation", "rfi"),
        )

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

    def load_manifest(self) -> Optional[Dict[str, Any]]:
        """
        Load metadata from the Sentinel-1 manifest.safe XML file.

        Returns:
            dict: Parsed metadata as a structured dictionary.
        """
        if not self._product_path:
            raise ValueError("Product path is not set.")
        manifest_path = os.path.join(self._product_path, "manifest.safe")
        try:
            tree = ET.parse(manifest_path)
            root = tree.getroot()
            metadata = {}

            platform = root.find(".//safe:platform", self._ns)
            if platform is not None:
                metadata["mission"] = extract_text(platform, "safe:familyName", self._ns) + extract_text(platform,
                                                                                                         "safe:number",
                                                                                                         self._ns)

            orbit_ref = root.find(".//safe:orbitReference", self._ns)
            if orbit_ref is not None:
                metadata["orbit_number"] = np.array(
                    [
                        extract_int(orbit_ref, 'safe:orbitNumber[@type="start"]', self._ns),
                        extract_int(orbit_ref, 'safe:orbitNumber[@type="stop"]', self._ns),
                    ]
                )
                metadata["relative_orbit_number"] = np.array(
                    [
                        extract_int(
                            orbit_ref,
                            'safe:relativeOrbitNumber[@type="start"]',
                            self._ns,
                        ),
                        extract_int(
                            orbit_ref,
                            'safe:relativeOrbitNumber[@type="stop"]',
                            self._ns,
                        ),
                    ]
                )
                metadata["cycle_number"] = extract_int(orbit_ref, "safe:cycleNumber", self._ns)
                metadata["ascending_node_time"] = extract_datetime(orbit_ref, "s1:ascendingNodeTime", self._ns)

            product_info = root.find(".//s1sarl1:standAloneProductInformation", self._ns)
            if product_info is not None:
                metadata["instrument_config"] = extract_int(product_info, "s1sarl1:instrumentConfigurationID", self._ns)
                metadata["mission_data_ID"] = extract_text(product_info, "s1sarl1:missionDataTakeID", self._ns)
                metadata["polarisation"] = [pol.text for pol in
                                            product_info.findall("s1sarl1:transmitterReceiverPolarisation", self._ns)]
                metadata["product_class"] = extract_text(product_info, "s1sarl1:productClass", self._ns)
                metadata["product_composition"] = extract_text(product_info, "s1sarl1:productComposition", self._ns)
                metadata["product_type"] = extract_text(product_info, "s1sarl1:productType", self._ns)
                metadata["product_timeliness"] = extract_text(product_info, "s1sarl1:productTimelinessCategory",
                                                              self._ns)
                metadata["slice_product_flag"] = extract_text(product_info, "s1sarl1:sliceProductFlag", self._ns)
                metadata["segment_start_time"] = extract_datetime(product_info, "s1sarl1:segmentStartTime", self._ns)
                metadata["slice_number"] = extract_int(product_info, "s1sarl1:sliceNumber", self._ns)
                metadata["total_slices"] = extract_int(product_info, "s1sarl1:totalSlices", self._ns)

            footprint = root.find(".//safe:footPrint", self._ns)
            if footprint is not None:
                coordinates = footprint.find("gml:coordinates", self._ns).text.split()
                lat = np.zeros(4)
                lon = np.zeros(4)
                for i, coord in enumerate(coordinates):
                    lat[i], lon[i] = map(float, coord.split(","))
                metadata["footprint"] = {"latitude": lat, "longitude": lon}

            metadata["band_order"] = ["HH", "HV"] if "HH" in metadata["polarisation"] else ["VV", "VH"]
            return metadata

        except Exception as e:
            warnings.warn(f"Error loading manifest data: {e}")
            return None

    def load_calibration(self, calibration_path: str) -> Optional[Dict[str, Any]]:
        """
        Load calibration data from Sentinel-1 calibration XML file.

        Args:
            calibration_path (str): Path to the calibration file.

        Returns:
            dict: Calibration table with constants and metadata.
        """
        try:
            tree = ET.parse(calibration_path)
            root = tree.getroot()

            info = {}
            ads_header = root.find("adsHeader")
            if ads_header is not None:
                for child in ads_header:
                    info[child.tag] = child.text
            else:
                warnings.warn("Warning: 'adsHeader' not found")

            cal_vectors = root.findall(".//calibrationVector")
            if not cal_vectors:
                warnings.warn("Warning: 'calibrationVectorList' not found")
                return None

            pixel = np.array(list(map(int, cal_vectors[0][2].text.split())))
            num_vectors = len(cal_vectors)

            azimuth_time = np.empty((num_vectors, len(pixel)), dtype="datetime64[us]")
            line = np.zeros(num_vectors, dtype=int)
            sigma_0 = np.zeros((num_vectors, len(pixel)), dtype=float)
            beta_0 = np.zeros((num_vectors, len(pixel)), dtype=float)
            gamma = np.zeros((num_vectors, len(pixel)), dtype=float)
            dn = np.zeros((num_vectors, len(pixel)), dtype=float)

            for i, vector in enumerate(cal_vectors):
                azimuth_time[i, :] = np.datetime64(vector.find("azimuthTime").text)
                line[i] = int(vector.find("line").text)
                sigma_0[i, :] = np.array(list(map(float, vector.find("sigmaNought").text.split())))
                beta_0[i, :] = np.array(list(map(float, vector.find("betaNought").text.split())))
                gamma[i, :] = np.array(list(map(float, vector.find("gamma").text.split())))
                dn[i, :] = np.array(list(map(float, vector.find("dn").text.split())))

            calibration_data = {
                "abs_calibration_const": float(root.find(".//absoluteCalibrationConstant").text),
                "row": line,
                "column": pixel,
                "azimuth_time": azimuth_time,
                "sigma_0": sigma_0,
                "beta_0": beta_0,
                "gamma": gamma,
                "dn": dn,
            }

            return calibration_data

        except Exception as e:
            warnings.warn(f"Error loading calibration data: {e}")
            return None

    def load_geotransform(self, annotation_path: str) -> Optional[Dict[str, Any]]:
        """
        Load geolocation annotation data from Sentinel-1 annotation XML file.

        Args:
            annotation_path (str): Path to the annotation file.

        Returns:
            dict: A dictionary with geolocation tie-points and metadata.
        """
        try:
            tree = ET.parse(annotation_path)
            root = tree.getroot()

            geo_points = root.findall(".//geolocationGridPoint")
            n_points = len(geo_points)

            azimuth_time = np.empty(n_points, dtype="datetime64[us]")
            slant_range_time = np.zeros(n_points, dtype=float)
            row = np.zeros(n_points, dtype=int)
            column = np.zeros(n_points, dtype=int)
            latitude = np.zeros(n_points, dtype=float)
            longitude = np.zeros(n_points, dtype=float)
            height = np.zeros(n_points, dtype=float)
            incidence_angle = np.zeros(n_points, dtype=float)
            elevation_angle = np.zeros(n_points, dtype=float)

            for i, point in enumerate(geo_points):
                azimuth_time[i] = np.datetime64(point.find("azimuthTime").text)
                slant_range_time[i] = float(point.find("slantRangeTime").text)
                row[i] = int(point.find("line").text)
                column[i] = int(point.find("pixel").text)
                latitude[i] = float(point.find("latitude").text)
                longitude[i] = float(point.find("longitude").text)
                height[i] = float(point.find("height").text)
                incidence_angle[i] = float(point.find("incidenceAngle").text)
                elevation_angle[i] = float(point.find("elevationAngle").text)

            geo_data = {
                "azimuth_time": azimuth_time,
                "slant_range_time": slant_range_time,
                "row": row,
                "column": column,
                "latitude": latitude,
                "longitude": longitude,
                "height": height,
                "incidence_angle": incidence_angle,
                "elevation_angle": elevation_angle,
            }

            return geo_data

        except Exception as e:
            warnings.warn(f"Error loading annotation data: {e}")
            return None

    def load_metadata(self, annotations_path):
        """
        Load and parse metadata from a Sentinel-1 annotation XML file.

        Args:
            annotations_path (str): Path to the annotation XML file.

        Returns:
            dict: Parsed metadata as a structured dictionary.
        """
        try:
            tree = ET.parse(annotations_path)
            root = tree.getroot()
            metadata = {}

            # Extracting adsHeader information
            # ads_header = root.find('adsHeader')

            # Extracting quality information
            quality_info = root.find("qualityInformation")
            if quality_info is not None:
                metadata["product_quality_index"] = extract_float(quality_info, "productQualityIndex", self._ns)
                quality_data = quality_info.find("qualityData")
                if quality_data is not None:
                    downlink_quality = quality_data.find("downlinkQuality")
                    if downlink_quality is not None:
                        metadata.update(
                            {
                                "i_input_data_mean": extract_float(downlink_quality, "iInputDataMean", self._ns),
                                "q_input_data_mean": extract_float(downlink_quality, "qInputDataMean", self._ns),
                                "num_downlink_input_data_gaps": extract_text(
                                    downlink_quality,
                                    "numDownlinkInputDataGaps",
                                    self._ns,
                                ),
                                "mean_pg_product_amplitude": extract_float(downlink_quality, "meanPgProductAmplitude",
                                                                           self._ns),
                                "std_dev_pg_product_amplitude": extract_float(
                                    downlink_quality,
                                    "stdDevPgProductAmplitude",
                                    self._ns,
                                ),
                            }
                        )

            # Extracting general product information
            product_info = root.find("generalAnnotation/productInformation")
            if product_info is not None:
                metadata.update(
                    {
                        "pass": extract_text(product_info, "pass", self._ns),
                        "platform_heading": extract_float(product_info, "platformHeading", self._ns),
                        "range_sampling_rate": extract_float(product_info, "rangeSamplingRate", self._ns),
                        "radar_frequency": extract_float(product_info, "radarFrequency", self._ns),
                        "azimuth_steering_rate": extract_float(product_info, "azimuthSteeringRate", self._ns),
                    }
                )

            # Extracting downlink information
            downlink_info_list = root.findall("generalAnnotation/downlinkInformationList/downlinkInformation")
            metadata["downlink_info"] = []
            for downlink_info in downlink_info_list:
                info = {
                    "swath": extract_text(downlink_info, "swath", self._ns),
                    "azimuth_time": extract_datetime(downlink_info, "azimuthTime", self._ns),
                    "prf": extract_float(downlink_info, "prf", self._ns),
                }
                metadata["downlink_info"].append(info)

            # Extracting orbit information
            orbit_list = root.findall("generalAnnotation/orbitList/orbit")
            metadata["orbit"] = []
            for orbit in orbit_list:
                orbit_data = {
                    "time": extract_datetime(orbit, "time", self._ns),
                    "position": {
                        "x": extract_float(orbit.find("position"), "x", self._ns),
                        "y": extract_float(orbit.find("position"), "y", self._ns),
                        "z": extract_float(orbit.find("position"), "z", self._ns),
                    },
                    "velocity": {
                        "x": extract_float(orbit.find("velocity"), "x", self._ns),
                        "y": extract_float(orbit.find("velocity"), "y", self._ns),
                        "z": extract_float(orbit.find("velocity"), "z", self._ns),
                    },
                }
                metadata["orbit"].append(orbit_data)

            # Extracting attitude information
            attitude_list = root.findall("generalAnnotation/attitudeList/attitude")
            # metadata['attitude'] = []

            return metadata

        except Exception as e:
            warnings.warn(f"Error loading metadata: {e}")
            return None

    def load_rfi_metadata(self, rfi_path):
        """
        Load and parse metadata from a Sentinel-1 RFI XML file.

        Args:
            rfi_path (str): Path to the RFI XML file.

        Returns:
            dict: Parsed RFI metadata as a structured dictionary.
        """

        try:
            tree = ET.parse(rfi_path)
            root = tree.getroot()
            metadata = {}

            # Extracting adsHeader information
            ads_header = root.find("adsHeader")
            if ads_header is not None:
                metadata.update(
                    {
                        "mission_id": extract_text(ads_header, "missionId", self._ns),
                        "product_type": extract_text(ads_header, "productType", self._ns),
                        "polarisation": extract_text(ads_header, "polarisation", self._ns),
                        "mode": extract_text(ads_header, "mode", self._ns),
                        "swath": extract_text(ads_header, "swath", self._ns),
                        "start_time": extract_datetime(ads_header, "startTime", self._ns),
                        "stop_time": extract_datetime(ads_header, "stopTime", self._ns),
                        "absolute_orbit_number": extract_text(ads_header, "absoluteOrbitNumber", self._ns),
                        "mission_data_take_id": extract_text(ads_header, "missionDataTakeId", self._ns),
                        "image_number": extract_text(ads_header, "imageNumber", self._ns),
                        "rfi_mitigation_applied": extract_text(root, "rfiMitigationApplied", self._ns),
                    }
                )

            # Extracting RFI detection from noise reports
            noise_reports = root.findall("rfiDetectionFromNoiseReportList/rfiDetectionFromNoiseReport")
            metadata["rfi_detection_from_noise"] = []
            for report in noise_reports:
                report_data = {
                    "swath": extract_text(report, "swath", self._ns),
                    "noise_sensing_time": extract_datetime(report, "noiseSensingTime", self._ns),
                    "rfi_detected": extract_text(report, "rfiDetected", self._ns),
                    "max_kl_divergence": extract_float(report, "maxKLDivergence", self._ns),
                    "max_fisher_z": extract_float(report, "maxFisherZ", self._ns),
                    "max_rfi_psd": extract_float(report, "maxRfiPsd", self._ns),
                }
                metadata["rfi_detection_from_noise"].append(report_data)

            # Extracting RFI burst reports
            burst_reports = root.findall("rfiBurstReportList/rfiBurstReport")
            metadata["rfi_burst_reports"] = []
            for burst in burst_reports:
                burst_data = {
                    "swath": extract_text(burst, "swath", self._ns),
                    "azimuth_time": extract_datetime(burst, "azimuthTime", self._ns),
                    "in_band_out_band_power_ratio": extract_float(burst, "inBandOutBandPowerRatio", self._ns),
                }
                metadata["rfi_burst_reports"].append(burst_data)

            metadata = Sentinel1MetadataLoader.filter_rfi_info(metadata)
            return metadata

        except:
            return {"RFI": "No RFI files"}

    def filter_rfi_info(rfi_metadata):
        """
        Filter and return RFI information if any RFI is detected.

        Args:
            rfi_metadata (dict): Parsed RFI metadata.

        Returns:
            dict: {"RFI": True, "details": [...]} if RFI is detected, else {"RFI": False}.
        """
        rfi_detected_info = []

        # Check for RFI detection in 'rfi_detection_from_noise'
        for report in rfi_metadata.get("rfi_detection_from_noise", []):
            if report["rfi_detected"] == "true":  # Only include if RFI was detected
                rfi_detected_info.append(
                    {
                        "swath": report["swath"],
                        "noise_sensing_time": report["noise_sensing_time"],
                        "max_kl_divergence": report["max_kl_divergence"],
                        "max_fisher_z": report["max_fisher_z"],
                        "max_rfi_psd": report["max_rfi_psd"],
                    }
                )

        # Check if any RFI was detected
        if rfi_detected_info:
            return {"RFI": True, "details": rfi_detected_info}
        else:
            return {"RFI": False}

    def validate_paths(self) -> bool:
        """Validate the existence of critical product files and directories."""
        required_paths = [
            os.path.join(self._product_path, "manifest.safe"),
            os.path.join(self._product_path, "annotation"),
            os.path.join(self._product_path, "annotation", "calibration"),
        ]
        for path in required_paths:
            if not os.path.exists(path):
                warnings.warn(f"Required path not found: {path}")
                return False
        return True

    @property
    def annotations_path(self) -> List[str]:
        """Retrieve annotation XML files."""
        return get_files(os.path.join(self._product_path, "annotation"))

    @annotations_path.setter
    def annotations_path(self, path: str):
        """Setter for product path."""

        self._annotations_path = path

    @property
    def calibration_path(self) -> List[str]:
        """Retrieve calibration XML files."""
        return get_files(
            os.path.join(self._product_path, "annotation", "calibration"),
            include="calibration",
        )

    @calibration_path.setter
    def calibration_path(self, path: str):
        """Setter for product path."""

        self._calibration_path = path

    @property
    def noise_path(self) -> List[str]:
        """Retrieve noise XML files."""
        return get_files(
            os.path.join(self._product_path, "annotation", "calibration"),
            include="noise",
        )

    @noise_path.setter
    def noise_path(self, path: str):
        """Setter for product path."""

        self._noise_path = path

    @property
    def rfi_path(self) -> List[str]:
        """Retrieve RFI XML files."""
        return get_files(os.path.join(self._product_path, "annotation", "rfi"))

    @rfi_path.setter
    def rfi_path(self, path: str):
        """Setter for product path."""

        self._rfi_path = path
