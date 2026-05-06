import datetime
import json
import os
import typing

import PIL.Image
import numpy
import rasterio
import xarray
from PIL.Image import Image

from satproducts.common.image_slice import ImageSlice
from satproducts.products.base.product import Product
from satproducts.transformers.base_transformer import BaseTransformer
from satproducts.transformers.transformer import Transformer


class LandsatProduct(Product):
    bands = ["coastal", "blue", "green", "red", "nir08", "swir16", "swir22", "pan", "cirrus", "lwir11", "lwir12"]

    def __init__(self, product_path):
        super().__init__(product_path=product_path)
        with open(os.path.join(product_path, "MTL.json")) as f:
            mtd = json.load(f)

        date = mtd["LANDSAT_METADATA_FILE"]["IMAGE_ATTRIBUTES"]["DATE_ACQUIRED"]
        timestamp = mtd["LANDSAT_METADATA_FILE"]["IMAGE_ATTRIBUTES"]["SCENE_CENTER_TIME"]
        self._timestamp = datetime.datetime.fromisoformat(date + "T" + timestamp)
        corners = mtd["LANDSAT_METADATA_FILE"]["PROJECTION_ATTRIBUTES"]
        self._footprint = {
            "type": "Polygon",
            "coordinates": [[
                [float(corners["CORNER_UL_LON_PRODUCT"]), float(corners["CORNER_UL_LAT_PRODUCT"])],
                [float(corners["CORNER_UR_LON_PRODUCT"]), float(corners["CORNER_UR_LAT_PRODUCT"])],
                [float(corners["CORNER_LR_LON_PRODUCT"]), float(corners["CORNER_LR_LAT_PRODUCT"])],
                [float(corners["CORNER_LL_LON_PRODUCT"]), float(corners["CORNER_LL_LAT_PRODUCT"])],
                [float(corners["CORNER_UL_LON_PRODUCT"]), float(corners["CORNER_UL_LAT_PRODUCT"])]
            ]]
        }
        das = [xarray.open_dataarray(os.path.join(product_path, b + ".TIF")).chunk() for b in self.bands]
        target_da = das[self.bands.index("pan")]
        reindexed_das = [da.reindex_like(target_da, method="nearest").chunk() for da in das]
        self._product = xarray.concat(reindexed_das, dim="band")

        rgb = self.product.isel(band=[3, 2, 1]).astype("float32")
        weights = numpy.array([0.3, 0.3, 0.35], dtype="float32")
        rgb *= weights[:, None, None]
        rgb /= rgb.sum(dim="band")
        rgb *= self.product.isel(band=7).astype("float32")
        rgb = (.75 * rgb.fillna(0) ** (1 / 1.4))
        self._tci = rgb.clip(0, 255).astype("uint8")

        self._width = self.product.sizes['x']
        self._height = self.product.sizes['y']
        self._channels = self.product.sizes['band']

        with rasterio.open(os.path.join(product_path, self.bands[7] + ".TIF")) as src:
            self._transformer = Transformer(transform=src.transform, crs=src.crs)

    def tile(self, tile_size) -> typing.Generator[typing.Tuple[ImageSlice, numpy.ndarray], None, None]:
        p = self.product.chunk(dict(x=tile_size, y=tile_size, band=-1))
        for image_slice in self.slices(tile_size):
            block = p.isel(band=[3, 2, 1, 7], **image_slice.to_slice()).to_numpy().astype(numpy.float32)
            rgb = block[0:3]
            pan = block[3]
            weights = numpy.array([0.3, 0.3, 0.35], dtype="float32")
            rgb *= weights[:, None, None]
            rgb /= (rgb.sum(axis=0) + 1e-12)
            rgb *= pan
            rgb = (.75 * numpy.nan_to_num(rgb) ** (1 / 1.4))
            yield image_slice, rgb.astype("uint8")

    def viewable(self, image_slice: ImageSlice = None) -> Image:
        tci = self._tci
        if image_slice:
            tci = tci[image_slice.to_slice()]
        tci = tci.transpose("y", "x", "band")
        return tci

    def view_thumbnail(self, *args, **kwargs) -> Image:
        image = PIL.Image.open(
            os.path.join(self.product_path, next(o for o in os.listdir(self.product_path) if o.endswith("-ql.jpg"))))
        return image

    @property
    def transformer(self, *args, **kwargs) -> BaseTransformer:
        return self._transformer

    @property
    def product(self):
        return self._product

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def channels(self):
        return self._channels

    @property
    def timestamp(self) -> datetime.datetime:
        return self._timestamp

    @property
    def footprint(self) -> dict:
        return self._footprint
