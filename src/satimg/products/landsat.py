"""Landsat Collection-2 Level-1 reader."""

from __future__ import annotations

import datetime
import json
import os
from functools import cached_property

import numpy
import rasterio

from satimg.product import Product
from satimg.raster import Raster
from satimg.readers import merge_bands
from satimg.registry import register
from satimg.transform import Transformer

#: band order of :attr:`LandsatProduct.raw`.
BANDS = ("coastal", "blue", "green", "red", "nir08", "swir16", "swir22", "pan",
         "cirrus", "lwir11", "lwir12")
_PAN = BANDS.index("pan")
_RGB = (BANDS.index("red"), BANDS.index("green"), BANDS.index("blue"))


@register(r"^LC(0[1-9])_L1(TP|GT)_\d+_\d+_\d+_\d+_T1$")
class LandsatProduct(Product):
    """``raw`` is every band resampled to the 15 m panchromatic grid; ``rgb`` is a
    pan-sharpened true-colour render."""

    def __init__(self, path: str):
        super().__init__(path)
        with open(os.path.join(path, "MTL.json")) as f:
            mtl = json.load(f)["LANDSAT_METADATA_FILE"]

        attrs = mtl["IMAGE_ATTRIBUTES"]
        self._timestamp = datetime.datetime.fromisoformat(
            f"{attrs['DATE_ACQUIRED']}T{attrs['SCENE_CENTER_TIME']}"
        )
        proj = mtl["PROJECTION_ATTRIBUTES"]
        ring = [
            (float(proj[f"CORNER_{c}_LON_PRODUCT"]), float(proj[f"CORNER_{c}_LAT_PRODUCT"]))
            for c in ("UL", "UR", "LR", "LL", "UL")
        ]
        self._footprint = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        }
        with rasterio.open(os.path.join(path, "pan.TIF")) as src:
            self._transformer = Transformer(src.transform, src.crs)

    @cached_property
    def raw(self) -> Raster:
        paths = [os.path.join(self._path, f"{b}.TIF") for b in BANDS]
        return Raster(merge_bands(paths, match=_PAN), transform=self._transformer, name="dn")

    def _render_rgb(self) -> Raster:
        a = self.raw.array
        rgb = a.isel(band=list(_RGB)).astype("float32")
        weights = numpy.array([0.3, 0.3, 0.35], dtype="float32")
        rgb = rgb * weights[:, None, None]
        rgb = rgb / (rgb.sum(dim="band") + 1e-12)
        rgb = rgb * a.isel(band=_PAN).astype("float32")
        rgb = 0.75 * rgb.fillna(0) ** (1 / 1.4)
        return Raster(rgb.clip(0, 255).astype("uint8"), transform=self._transformer, name="rgb")

    @property
    def transformer(self) -> Transformer:
        return self._transformer

    @property
    def timestamp(self) -> datetime.datetime:
        return self._timestamp

    @property
    def footprint(self) -> dict:
        return self._footprint

    def thumbnail(self):
        import PIL.Image

        for suffix in ("_small.jpeg", "thumbnail.jpeg", ".jpeg"):
            match = next((o for o in os.listdir(self._path) if o.endswith(suffix)), None)
            if match:
                return PIL.Image.open(os.path.join(self._path, match))
        raise FileNotFoundError(f"no thumbnail under {self._path}")
