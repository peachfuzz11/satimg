import abc
import json
import logging
import os
import typing
from enum import Enum

import PIL
import rioxarray
from PIL.Image import Image

from satproducts.common.image_slice import ImageSlice
from satproducts.transformers.base_transformer import BaseTransformer
from satproducts.products import RasterProduct


class UmbraType(Enum):
    GEC = "GEC"
    SIDD = "SIDD"
    SICD = "SICD"


class UmbraProduct(RasterProduct, abc.ABC):
    def __init__(self, product_path):
        super().__init__(product_path)
        self._umbra_type = UmbraType.SICD  # Default value is GEC

    def view_thumbnail(self, *args, **kwargs) -> Image:
        raise NotImplementedError("Thumbnail view is not implemented for Umbra products.")

    def view_full(self, band: typing.Union[int, str] = 0, stride: int = 1, *args, **kwargs) -> Image:
        """
        View the full image based on the selected band and stride.

        Parameters:
        - band: (int or str) The band to visualize. Can be 0, 1, or "RGB".
        - stride: (int) The stride size for stepping through the image pixels. Defaults to 1 (no skipping).
        """
        assert band in [0, 1, "RGB"], f"Invalid band: {band}. Must be 0, 1, or 'RGB'."

        with self:
            # Handle the band selection and apply stride lazily
            if band == "RGB":
                raise NotImplementedError  # Implement RGB logic if needed
            else:
                # Apply the stride to x and y, and select the band
                downsampled_image_data = self.product.isel(x=slice(0, None, stride), y=slice(0, None, stride),
                                                           band=band)  # Select a specific band

            # Load the data into memory after applying stride
            downsampled_image_data = downsampled_image_data.load()
            try:
                downsampled_image_data = downsampled_image_data.to_array().data
            except:
                downsampled_image_data = downsampled_image_data.to_numpy()

            # Check if the array has more than 2 dimensions (e.g., multi-band)
            if downsampled_image_data.ndim == 3:
                # Select the first channel/band if it's a multi-dimensional array
                downsampled_image_data = downsampled_image_data[0]

            # Normalize the image data and convert it to uint8
            downsampled_image_data = self._normalize_fn(downsampled_image_data).astype("uint8")

            # Convert the array to a PIL image (grayscale mode "L")
            image = PIL.Image.fromarray(downsampled_image_data, mode="L")

        return image

    def view_quicklook(self, *args, **kwargs) -> Image:
        raise NotImplementedError("Quicklook view is not implemented for Umbra products.")

    def view_slice(self, image_slice: ImageSlice, band: typing.Union[int, str] = 0) -> Image:
        # Band can be int(0), int(1), or str(RGB).
        assert band in [0, 1, "RGB"], f"Invalid band: {band}. Must be 0, 1, or 'RGB'."

        with self:
            image = self.read_slice(image_slice)
            if band == "RGB":
                raise NotImplementedError
            else:
                image = image[band]
            image = PIL.Image.fromarray(self._normalize_fn(image).astype("uint8"))
        return image

    def open(self, *args, **kwargs):
        try:
            self.files = self._locate_files()
        except Exception as e:
            logging.error(f"Failed to locate RCM files: {e}")
            raise
        try:
            self._product = self._load_bands()
        except Exception as e:
            logging.error(f"Failed to load RCM bands: {e}")
            raise

    def _init(self):
        with self:
            try:
                self._width = self.product.sizes["x"]
                self._height = self.product.sizes["y"]
                self._nr_bands = self.product.sizes["band"]

                self._extract_metadata()

                self.validate()
            except Exception as e:
                logging.error(f"Failed to initialize RCM product: {e}")
                raise

    def get_transformer(self, **kwargs) -> BaseTransformer:
        return super().get_transformer(self.files["tif"][0])

    def _locate_files(self):
        """Automatically locate the Umbra .tif, .nitf, .json, and other important files."""
        files_dict = {
            "tif": [],
            "SICD": [],
            "SIDD": [],
            "json": [],
            "cphd": [],
            "other_files": [],
        }

        for root, dirs, files in os.walk(self._product_path):
            for file in files:
                file_path = os.path.join(root, file)

                if file.endswith(".tif"):
                    files_dict["tif"].append(file_path)
                elif file.endswith(".nitf") and ("SIDD" in file):
                    files_dict["SIDD"].append(file_path)
                elif file.endswith(".nitf") and ("SICD" in file):
                    files_dict["SICD"].append(file_path)
                elif file.endswith(".json") and "METADATA" in file:
                    files_dict["json"].append(file_path)
                elif file.endswith(".cphd") and "CPHD" in file:
                    files_dict["cphd"].append(file_path)
                else:
                    files_dict["other_files"].append(file_path)

        return files_dict

    def _load_bands(self):
        """Load multiple .tif files as SAR bands and concatenate them along the 'band' dimension."""

        if self.umbra_type == UmbraType.GEC:
            files = self.files["tif"]
        elif self.umbra_type == UmbraType.SIDD:
            files = self.files["SIDD"]
        elif self.umbra_type == UmbraType.SICD:
            files = self.files["SICD"]

        if len(files) == 1:
            return rioxarray.open_rasterio(files[0], chunks=True)

        else:
            raise NotImplementedError("Loading multiple bands is not implemented yet.")

    @property
    def umbra_type(self) -> UmbraType:
        return self._umbra_type

    @umbra_type.setter
    def umbra_type(self, value: UmbraType):
        if not isinstance(value, UmbraType):
            raise ValueError(f"Invalid umbra type: {value}. Must be one of {list(UmbraType)}.")
        self._umbra_type = value

    def parse_metadata(self):
        """Extract transformers from the JSON file."""
        json_file = self.files["json"][0]  # Get the first JSON file from the list

        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            collects = data.get("collects", [{}])[0]
            derived_products = data.get("derivedProducts", {})

            self.metadata = {
                "file_path": json_file,
                "version": data.get("version"),
                "vendor": data.get("vendor"),
                "imagingMode": data.get("imagingMode"),
                "orderType": data.get("orderType"),
                "productSku": data.get("productSku"),
                "baseIpr": data.get("baseIpr"),
                "targetIpr": data.get("targetIpr"),
                "satellite": data.get("umbraSatelliteName"),
                "startAtUTC": collects.get("startAtUTC"),
                "endAtUTC": collects.get("endAtUTC"),
                "radarBand": collects.get("radarBand"),
                "radarCenterFrequencyHz": collects.get("radarCenterFrequencyHz"),
                "polarizations": collects.get("polarizations"),
                "angleAzimuthDegrees": collects.get("angleAzimuthDegrees"),
                "angleGrazingDegrees": collects.get("angleGrazingDegrees"),
                "angleIncidenceDegrees": collects.get("angleIncidenceDegrees"),
                "angleSquintDegrees": collects.get("angleSquintDegrees"),
                "slantRangeMeters": collects.get("slantRangeMeters"),
                "satelliteTrack": collects.get("satelliteTrack"),
                "observationDirection": collects.get("observationDirection"),
                "sceneCenterPointLla": collects.get("sceneCenterPointLla"),
                "maxGroundResolution": collects.get("maxGroundResolution"),
                "GEC_numRows": derived_products.get("GEC", [{}])[0].get("numRows"),
                "GEC_numColumns": derived_products.get("GEC", [{}])[0].get("numColumns"),
                "GEC_groundResolution": derived_products.get("GEC", [{}])[0].get("groundResolution"),
                "SICD_numRows": derived_products.get("SICD", [{}])[0].get("numRows"),
                "SICD_numColumns": derived_products.get("SICD", [{}])[0].get("numColumns"),
                "SICD_groundResolution": derived_products.get("SICD", [{}])[0].get("groundResolution"),
                "SICD_slantResolution": derived_products.get("SICD", [{}])[0].get("slantResolution"),
            }

        except Exception as e:
            logging.error(f"Failed to extract transformers from {json_file}: {e}")
            raise

    def _extract_metadata(self):
        self.parse_metadata()

    def validate(self):
        pass
