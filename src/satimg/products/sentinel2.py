"""Sentinel-2 Level-1C reader."""

from __future__ import annotations

import datetime
import os
from functools import cached_property
from xml.etree import ElementTree

import rasterio
import xarray

from satimg.product import Product
from satimg.raster import Raster
from satimg.readers import find_file, merge_bands, open_band
from satimg.registry import register
from satimg.transform import Transformer


def _parse_mtd(path: str) -> dict:
    mtd = find_file(path, "MTD_MSIL1C.xml")
    if mtd is None:
        raise FileNotFoundError(f"no MTD_MSIL1C.xml under {path}")
    root = ElementTree.parse(mtd).getroot()

    physical = {}
    for info in root.findall(".//Spectral_Information"):
        physical[int(info.get("bandId"))] = info.get("physicalBand")

    reflectance, tci = [], None
    for band_id, elem in enumerate(root.iter("IMAGE_FILE")):
        file_path = os.path.join(path, elem.text) + ".jp2"
        if physical.get(band_id) is None:
            tci = file_path
        else:
            reflectance.append(file_path)

    ext = [float(v) for v in root.find(".//EXT_POS_LIST").text.split()]
    ring = [(ext[i + 1], ext[i]) for i in range(0, len(ext), 2)]
    return {
        "reflectance": reflectance,
        "tci": tci,
        "footprint": {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        },
    }


def _sensing_time(path: str) -> datetime.datetime:
    tl = find_file(path, "MTD_TL.xml")
    root = ElementTree.parse(tl).getroot()
    return datetime.datetime.strptime(
        root.find(".//SENSING_TIME").text, "%Y-%m-%dT%H:%M:%S.%fZ"
    )


@register(r"^S2[ABCD]_MSIL1C_\d{8}T\d{6}_.*_\d{8}T\d{6}\.SAFE$")
class Sentinel2L1CProduct(Product):
    """``raw`` is the 13 reflectance bands resampled to the 10 m grid; ``rgb`` is
    the shipped True Colour Image."""

    def __init__(self, path: str):
        super().__init__(path)
        self._meta = _parse_mtd(path)
        self._timestamp = _sensing_time(path)
        with rasterio.open(self._meta["reflectance"][1]) as src:
            self._transformer = Transformer(src.transform, src.crs)

    @cached_property
    def raw(self) -> Raster:
        da = merge_bands(self._meta["reflectance"], match=1)
        return Raster(da, transform=self._transformer, name="reflectance")

    def _render_rgb(self) -> Raster:
        tci = open_band(self._meta["tci"]).astype("uint8")
        return Raster(tci, transform=self._transformer, name="tci")

    @property
    def transformer(self) -> Transformer:
        return self._transformer

    @property
    def timestamp(self) -> datetime.datetime:
        return self._timestamp

    @property
    def footprint(self) -> dict:
        return self._meta["footprint"]

    def thumbnail(self):
        import PIL.Image

        name = next(o for o in os.listdir(self._path) if o.endswith("-ql.jpg"))
        return PIL.Image.open(os.path.join(self._path, name))
