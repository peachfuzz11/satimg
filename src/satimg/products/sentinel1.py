"""Sentinel-1 GRD (IW and EW) reader."""

from __future__ import annotations

import datetime
import os
import re
from functools import cached_property
from xml.etree import ElementTree

import numpy
import PIL.Image
import rasterio

from satimg.product import Product
from satimg.raster import Raster
from satimg.readers import find_file, merge_bands
from satimg.registry import register
from satimg.transform import GCPTransformer

_NS = {
    "safe": "http://www.esa.int/safe/sentinel-1.0",
    "gml": "http://www.opengis.net/gml",
}


def _read_manifest(path: str) -> dict:
    manifest = find_file(path, "manifest.safe")
    if manifest is None:
        raise FileNotFoundError(f"no manifest.safe under {path}")
    root = ElementTree.parse(manifest).getroot()
    start = datetime.datetime.strptime(
        root.find(".//safe:startTime", _NS).text, "%Y-%m-%dT%H:%M:%S.%f"
    )
    stop = datetime.datetime.strptime(
        root.find(".//safe:stopTime", _NS).text, "%Y-%m-%dT%H:%M:%S.%f"
    )
    coords = [
        (float(lon), float(lat))
        for pair in root.find(".//gml:coordinates", _NS).text.split()
        for lat, lon in [pair.split(",")]
    ]
    return {
        "timestamp": start + (stop - start) / 2,
        "footprint": {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        },
    }


@register(r"^S1[ABCD]_(IW_GRDH|EW_GRDM)_1SD[HV]_\d{8}T\d{6}_\d{8}T\d{6}.*\.SAFE$")
class Sentinel1Product(Product):
    """Ground-range-detected Sentinel-1, either acquisition mode.

    ``raw`` holds the calibrated backscatter (one band per polarisation);
    ``rgb`` is the usual dB -> sigmoid stretch, greyscale repeated to 3 bands.
    """

    def __init__(self, path: str):
        super().__init__(path)
        meta = _read_manifest(path)
        self._timestamp = meta["timestamp"]
        self._footprint = meta["footprint"]
        with rasterio.open(path) as src:
            gcps, crs = src.gcps
        self._transformer = GCPTransformer(gcps, crs)

    @property
    def mode(self) -> str:
        m = re.search(r"_(IW|EW)_", os.path.basename(self._path))
        return m.group(1) if m else "IW"

    @cached_property
    def raw(self) -> Raster:
        measurement = os.path.join(self._path, "measurement")
        files = sorted(
            (os.path.join(measurement, f) for f in os.listdir(measurement)),
            key=lambda f: f[::-1],
        )
        return Raster(merge_bands(files), transform=self._transformer, name="sigma0")

    def _render_rgb(self) -> Raster:
        a = self.raw.array
        db = (10 * numpy.log10(a.where(a > 0))).fillna(0).mean("band")
        u8 = (255 / (1 + numpy.exp(-((db - 20) * 0.18)))).clip(0, 255).astype("uint8")
        rgb = u8.expand_dims(band=[0, 1, 2])
        return Raster(rgb, transform=self._transformer, name="rgb")

    @property
    def transformer(self) -> GCPTransformer:
        return self._transformer

    @property
    def timestamp(self) -> datetime.datetime:
        return self._timestamp

    @property
    def footprint(self) -> dict:
        return self._footprint

    def thumbnail(self) -> PIL.Image.Image:
        for name in ("thumbnail.png", "quick-look.png"):
            p = os.path.join(self._path, "preview", name)
            if os.path.isfile(p):
                return PIL.Image.open(p)
        raise FileNotFoundError(f"no preview image under {self._path}")
