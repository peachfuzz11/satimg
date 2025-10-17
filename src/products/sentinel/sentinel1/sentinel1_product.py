import abc
import os

import PIL.Image
import rasterio
import rioxarray
from PIL.Image import Image

from common.image_slice import ImageSlice
from metadata.gcp_transformer import GCPTransformer
from products.base.image.raster_product import RasterProduct
from products.sentinel.sentinel1.sentinel1_metadata import Sentinel1MetadataLoader
from products.sentinel.sentinel1.service import sentinel1_service


class Sentinel1Product(RasterProduct, abc.ABC):
    def __init__(self, product_path):
        super().__init__(product_path)
        self._extract_metadata()
        manifest_data = sentinel1_service.get_manifest_data(self.product_path)
        self._footprint = manifest_data["footprint"]
        self._timestamp = manifest_data["timestamp"]
        with self:
            self._width = self.product.sizes["x"]
            self._height = self.product.sizes["y"]

    def view_thumbnail(self, *args, **kwargs) -> Image:
        return self.view_quicklook(*args, **kwargs).resize((200, 200))

    def view_full(self, *args, **kwargs) -> Image:
        normalized_image = self._normalize_fn(self.product.values[1, :, :]).astype("uint8")
        # Convert the array to a PIL image (grayscale mode "L")
        image = PIL.Image.fromarray(normalized_image, mode="L")
        return image

    def view_quicklook(self, *args, **kwargs) -> Image:
        p = os.path.join(self.product_path, "preview", "thumbnail.png")
        if not os.path.isfile(p):
            p = os.path.join(self.product_path, "preview", "quick-look.png")
        return PIL.Image.open(p)

    def view_slice(self, image_slice: ImageSlice) -> Image:
        image = self.read_slice(image_slice).values[1, :, :]
        image = PIL.Image.fromarray(self._normalize_fn(image).astype("uint8"))
        return image

    def get_transformer(self, *args, **kwargs) -> GCPTransformer:
        with rasterio.open(self.product_path) as src:
            gcps, crs = src.gcps
            transformer = GCPTransformer(gcps=gcps, crs=crs)
        return transformer

    def _extract_metadata(self):
        """Extract metadata specific to Sentinel-1."""
        with rioxarray.open_rasterio(self.product_path, chunks={"band": 2, "y": 256, "x": 256}) as product:
            product_type = "GRD" if "GRD" in self._product_path else "SLC" if "SLC" in self._product_path else "RAW"
            self.metadata = {
                "file_path": self._product_path,
                "ProductType": product_type,
                "Satellite": product.attrs.get("MISSION_ID", "Sentinel-1"),
                "Swath": product.attrs.get("SWATH", "IW"),
                "Start": product.attrs.get("ACQUISITION_START_TIME", "Unknown"),
                "End": product.attrs.get("ACQUISITION_STOP_TIME", "Unknown"),
                "Shape": product.rio.shape,
                "resolution": product.attrs.get("PIXEL_SPACING", "Unknown"),
                "Mode": product.attrs.get("BEAM_MODE", "Unknown"),
                "Orbit": product.attrs.get("ORBIT_NUMBER", "Unknown"),
            }
            metadataClass = Sentinel1MetadataLoader(self._product_path)
            metadata = metadataClass.load_manifest()
            calibration_data = metadataClass.load_calibration(metadataClass.calibration_path[0])
            advanced_metadata = metadataClass.load_metadata(metadataClass.annotations_path[0])
            rfi_data = metadataClass.load_rfi_metadata(metadataClass.rfi_path[0])
            geotransform_data = metadataClass.load_geotransform(metadataClass.annotations_path[0])

            self.metadata.update(metadata)
            self.geotransform = geotransform_data
            self.calibration_data = calibration_data
            self.advanced_metadata = advanced_metadata
            self.rfi = rfi_data

    def __enter__(self):
        product = sentinel1_service.open_dataarrays(self.product_path)
        self._product = product
