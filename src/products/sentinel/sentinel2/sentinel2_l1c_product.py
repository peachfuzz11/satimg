import gc
import os

import PIL.Image
import rasterio
import xarray
from PIL.Image import Image

from common.image_slice import ImageSlice
from metadata.transformer import Transformer
from products.base.image.raster_product import RasterProduct
from products.sentinel.sentinel2.service import sentinel2_service


class Sentinel2L1CProduct(RasterProduct):
    def view_thumbnail(self, *args, **kwargs) -> Image:
        image = self.view_quicklook()
        image = image.resize((200, 200))
        return image

    def view_full(self, *args, **kwargs) -> Image:
        tci = [b["file_path"] for b in self._bands if b["physical_band"] is None][0]
        with xarray.open_dataarray(tci) as src:
            tci = src.transpose("y", "x", "band").values.astype("uint8")
        tci = PIL.Image.fromarray(tci)
        return tci

    def view_quicklook(self, *args, **kwargs) -> Image:
        image = PIL.Image.open(
            os.path.join(self.product_path, next(o for o in os.listdir(self.product_path) if o.endswith("-ql.jpg"))))
        return image

    def view_slice(self, image_slice: ImageSlice) -> Image:
        subset = self.tci[image_slice.to_slice()]
        tci = subset.transpose("y", "x", "band").values.astype("uint8")
        tci = PIL.Image.fromarray(tci)
        return tci

    def get_transformer(self, *args, **kwargs) -> Transformer:
        with rasterio.open(self._get_reference_band()['file_path']) as src:
            transformer = Transformer(transform=src.transform, crs=src.crs)
        return transformer

    def _get_reference_band(self):
        return self._bands[1]

    def __init__(self, product_path):
        super().__init__(product_path)
        mtd_msil1c_data = sentinel2_service.get_mtd_msil1c_data(product_path)
        bands = mtd_msil1c_data["bands"]
        self._footprint = mtd_msil1c_data["footprint"]
        mtd_tl_data = sentinel2_service.get_mtd_tl_data(self.product_path)
        self._timestamp = mtd_tl_data["timestamp"]
        self._bands = bands
        self._tci = None
        with self:
            self._width = self.product.sizes['x']
            self._height = self.product.sizes['y']
            self._nr_bands = self.product.sizes['band']

    def __enter__(self):
        das = sentinel2_service.open_dataarrays(self._bands)
        self._product = sentinel2_service.reindex_and_concatenate_dataarrays(das)
        tci = [b["file_path"] for b in self._bands if b["physical_band"] is None][0]
        with xarray.open_dataarray(tci) as src:
            self._tci = src

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self._product
        del self._tci
        gc.collect()

    @property
    def tci(self):
        return self._tci
