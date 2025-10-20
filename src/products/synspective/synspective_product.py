import abc
import gc

import PIL.Image
from PIL.Image import Image

from common.image_slice import ImageSlice
from products.base.image.raster_product import RasterProduct


class SynspectiveProduct(RasterProduct, abc.ABC):
    """
    Syspective GRD files comes in folders with an .xml and .tif file
    Syspective SLC files comes in folders with an .jpef and .nitf file
    """

    def __init__(self, product_path):
        super().__init__(product_path)

    def _init(self):
        with self:
            self._width = self.product.sizes["x"]
            self._height = self.product.sizes["y"]
            self._nr_bands = self.product.sizes["band"]

            self._extract_metadata()
            self.validate()

    def view_thumbnail(self, *args, **kwargs) -> Image:
        # raise not imprmented error with a costom text
        raise NotImplementedError("Synspective STRIX products does not have a thumbnail")

    def view_full(self, stride: int = 1, *args, **kwargs) -> Image:
        """
        View the full image based on the selected band and stride.

        Parameters:
        - band: (int or str) The band to visualize. Can be 0, 1, or "RGB".
        - stride: (int) The stride size for stepping through the image pixels. Defaults to 1 (no skipping).
        """

        with self:
            # Handle the band selection and apply stride lazily
            downsampled_image_data = self.product.isel(x=slice(0, None, stride),
                                                       y=slice(0, None, stride))  # Select data along x and y dimensions
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
        raise NotImplementedError("Synspective STRIX products does not have a quicklook")

    def view_slice(self, image_slice: ImageSlice) -> Image:
        with self:
            image = self.read_slice(image_slice)
            image = PIL.Image.fromarray(self._normalize_fn(image[0]).astype("uint8"))
        return image

    def _extract_metadata(self):
        """Extract transformers specific to Sentinel-1."""
        pass

    @abc.abstractmethod
    def validate(self):
        """
        general for Sentinel1.
        IW/EW etc is implemented in the subclass
        """
        test_slice = ImageSlice(0, 0, 5, 5)
        if not self._product_path:
            self._valid = False
            raise ValueError("No product path provided for Sentinel-1 IW product.")

        if self._product is None:
            self._valid = False
            raise ValueError("Dataset not loaded. Please call open() to load the product.")

        try:
            self._product.load()
        except Exception as e:
            self._valid = False
            raise ValueError(f"Failed to load dataset using rioxarray: {e}")

        if self._nr_bands != 1:
            self._valid = False
            raise ValueError(f"Invalid number of bands. Expected 1 but got {self._nr_bands}.")

        if self._height < 5 or self._width < 5:
            self._valid = False
            raise ValueError(
                f"Dataset dimensions are too small. Must be at least 5x5 pixels, but got {self._height}x{self._width}.")
        try:
            slice = self.read_slice(test_slice)
            del slice
            gc.collect()
        except Exception as e:
            self._valid = False
            raise ValueError(f"Failed to read a small slice from the dataset: {e}")

        # Define an image slice with specific window dimensions (modify this as per your data)
