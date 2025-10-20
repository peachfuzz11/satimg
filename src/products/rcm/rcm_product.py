import os
import typing
import xml.etree.ElementTree as ET

import PIL
import rioxarray
import xarray as xr
from PIL.Image import Image

from common.image_slice import ImageSlice
from transformers.base_transformer import BaseTransformer
from products.base.image.raster_product import RasterProduct
from products.rcm.rcm_metadata import RCMMetadataLoader


class RCMProduct(RasterProduct):
    def _init(self):
        with self:
            meta = RCMMetadataLoader(product_path=self._product_path)
            meta.load_product_metadata()

            self._width = self.product.sizes["x"]
            self._height = self.product.sizes["y"]
            self._nr_bands = self.product.sizes["band"]
            self.validate()

            # meta.load_calibration_incidence_angles()
            # meta.load_all_luts()
            # self._extract_metadata()

            self.metadata = meta.metadata
            self.advanced_metadata = meta.advanced_metadata

    def view_thumbnail(self, *args, **kwargs) -> Image:
        return self.view_quicklook().resize((200, 200))

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
        return PIL.Image.open(os.path.join(self.product_path, "preview", "productOverview.png"))

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
        """
        we use the tif files several times, so load it once and save it instead of using, e.g., rioxarray.open_rasterio(os.path.join(folder, o), chunks=True) for o in os.listdir(folder) if o.endswith(".tif")
        """

        tiff_files = []
        for root, dirs, files in os.walk(self._product_path):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".tif") and "imagery" in root and not file.startswith("."):
                    tiff_files.append(file_path)
        self.tiff_files = sorted(tiff_files)
        self._product = self._load_rcm_bands(self.tiff_files)
        if self.metadata is not None:
            if "VV" in self.metadata["band_order"]:
                try:
                    self._product = self._product.sel(band=["VV", "VH"])
                    self.metadata["band_order"] = ["VV", "VH"]
                    self.metadata["polarisation"] = ["VV", "VH"]
                except:
                    try:
                        self._product = self._product.sel(band=["VV"])
                        self.metadata["band_order"] = ["VV"]
                        self.metadata["polarisation"] = ["VV"]

                    except:
                        raise ValueError("Cant find VV in dataset")
            elif "HH" in self.metadata["band_order"]:
                try:
                    self._product = self._product.sel(band=["HH", "HV"])
                    self.metadata["band_order"] = ["HH", "HV"]
                    self.metadata["polarisation"] = ["HH", "HV"]
                except:
                    try:
                        self._product = self._product.sel(band=["HH"])
                        self.metadata["band_order"] = ["HH"]
                        self.metadata["polarisation"] = ["HH"]
                    except:
                        raise ValueError("Cant find HH in dataset")
            else:
                pass

    def get_transformer(self, *args, **kwargs) -> BaseTransformer:
        return super().get_transformer(self.tiff_files[0])

    def parse_metadata(self):
        """Extract transformers from the XML file."""
        xml_file = self.files["xml"][0]  # Get the first XML file from the list

        tree = ET.parse(xml_file)
        root = tree.getroot()

        ns = {"ns": "rcmGsProductSchema"}  # Define the XML namespace

        # Extract relevant transformers fields from the XML structure
        self.metadata = {
            "ProductType": root.find("ns:sourceAttributes/ns:radarParameters/ns:acquisitionType", ns).text,
            "productId": root.find("ns:productId", ns).text,
            "productApplication": root.find("ns:productApplication", ns).text,
            "documentIdentifier": root.find("ns:documentIdentifier", ns).text,
            "securityClassification": root.find("ns:securityAttributes/ns:securityClassification", ns).text,
            "specialHandlingRequired": root.find("ns:securityAttributes/ns:specialHandlingRequired", ns).text,
            "Satellite": root.find("ns:sourceAttributes/ns:satellite", ns).text,
            "sensor": root.find("ns:sourceAttributes/ns:sensor", ns).text,
            "polarizationDataMode": root.find("ns:sourceAttributes/ns:polarizationDataMode", ns).text,
            "beamMode": root.find("ns:sourceAttributes/ns:beamMode", ns).text,
            "beamModeMnemonic": root.find("ns:sourceAttributes/ns:beamModeMnemonic", ns).text,
            "rawDataStartTime": root.find("ns:sourceAttributes/ns:rawDataStartTime", ns).text,
            "polarizations": root.find("ns:sourceAttributes/ns:radarParameters/ns:polarizations", ns).text,
            "beams": root.find("ns:sourceAttributes/ns:radarParameters/ns:beams", ns).text,
        }

    def _extract_metadata(self):
        self.parse_metadata()

    @staticmethod
    def _load_rcm_bands(tif_files: list):
        """Load multiple .tif files as RCM bands and stack them into a single image.
        a Dataset is returned, instead of data arrays,.
        """
        band_data = []
        band_order = []  # List to store band names for assigning as coordinates
        all_attributes = {}  # Dictionary to store all unique attributes
        transform = None
        crs = None

        for i, tif_file in enumerate(tif_files):
            try:
                # Load the band using rioxarray
                band = rioxarray.open_rasterio(tif_file, chunks=True, masked=True)

                # Extract the band name from TIFF transformers
                band_name = band.attrs.get("TIFFTAG_IMAGEDESCRIPTION", f"band_{i + 1}")
                band_order.append(band_name)

                # Collect attributes for this band
                for key, value in band.attrs.items():
                    if key not in all_attributes:
                        all_attributes[key] = [value]
                    else:
                        all_attributes[key].append(value)

                # If 'band' dimension already exists, skip expanding it
                if "band" not in band.dims:
                    band = band.expand_dims(dim="band").assign_coords(band=[i + 1])

                band_data.append(band)

                # Extract transform and CRS from the first band
                if i == 0:
                    transform = band.rio.transform()
                    crs = band.rio.crs

            except Exception as e:
                raise ValueError(f"Failed to load {tif_file}: {e}")

        # Concatenate all bands along the 'band' dimension
        combined = xr.concat(band_data, dim="band")

        # Assign the band names as coordinates
        combined = combined.assign_coords(band=band_order)

        # Create a dataset with a single variable 'AMPLITUDE' similar to what we do for Sentinel-1.
        dataset = xr.Dataset({"AMPLITUDE": combined})

        # Set the same transform and CRS for all bands
        dataset = dataset.rio.write_transform(transform)
        dataset = dataset.rio.write_crs(crs)

        # Attach each attribute as an individual key in the dataset's attributes
        for key, values in all_attributes.items():
            dataset.attrs[key] = values

        return dataset
