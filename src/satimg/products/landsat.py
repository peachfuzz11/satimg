"""Landsat Collection-2 Level-1 reader."""

from __future__ import annotations

import datetime
import json
import os
from functools import cached_property

import numpy
import rasterio
import xarray
from rasterio.transform import Affine

from satimg.metadata import Metadata
from satimg.product import Product
from satimg.registry import register
from satimg.tiling import as_band_yx, label_bands
from satimg.transform import Transformer

#: band order of :attr:`LandsatProduct.raw`.
BANDS = ("coastal", "blue", "green", "red", "nir08", "swir16", "swir22", "pan",
         "cirrus", "lwir11", "lwir12")
_PAN = BANDS.index("pan")
_RGB = (BANDS.index("red"), BANDS.index("green"), BANDS.index("blue"))

#: per-pixel angle rasters (int16, hundredths of a degree) -> metadata field name.
_ANGLE_FILES = {
    "sun_zenith": "SZA.TIF",
    "sun_azimuth": "SAA.TIF",
    "view_zenith": "VZA.TIF",
    "view_azimuth": "VAA.TIF",
}
_ANGLE_SCALE = 0.01


@register(r"^LC(0[1-9])_L1(TP|GT)_\d+_\d+_\d+_\d+_T1$")
class LandsatProduct(Product):
    """``raw`` is every band resampled to the 15 m panchromatic grid; ``visual`` is
    a 3-band pan-sharpened true-colour render. ``metadata`` carries the per-pixel
    angle rasters (``sun_zenith``, ``sun_azimuth``, ``view_zenith``,
    ``view_azimuth``), read decimated to a coarse grid."""

    def __init__(self, path: str):
        super().__init__(path)
        with open(os.path.join(path, "MTL.json")) as f:
            mtl = json.load(f)["LANDSAT_METADATA_FILE"]

        attrs = mtl["IMAGE_ATTRIBUTES"]
        self._image_attrs = attrs
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
    def raw(self) -> xarray.DataArray:
        paths = [os.path.join(self._path, f"{b}.TIF") for b in BANDS]
        merged = self._merge_bands(paths, match=_PAN)
        return label_bands(as_band_yx(merged), BANDS)

    def _render_visual(self) -> xarray.DataArray:
        a = self.raw.drop_vars("band")  # positional indexing below; relabel at the end
        rgb = a.isel(band=list(_RGB)).astype("float32")
        weights = numpy.array([0.3, 0.3, 0.35], dtype="float32")
        rgb = rgb * weights[:, None, None]
        rgb = rgb / (rgb.sum(dim="band") + 1e-12)
        rgb = rgb * a.isel(band=_PAN).astype("float32")
        rgb = 0.75 * rgb.fillna(0) ** (1 / 1.4)
        return label_bands(as_band_yx(rgb.clip(0, 255).astype("uint8")),
                           ("red", "green", "blue"))

    def _read_metadata(self) -> Metadata:
        pan = self._transformer._transform
        rows = cols = None
        fields = {}
        for name, fname in _ANGLE_FILES.items():
            path = os.path.join(self._path, fname)
            with rasterio.open(path) as src:
                stride = max(1, max(src.height, src.width) // 256)
                out_h, out_w = src.height // stride, src.width // stride
                values = src.read(1, out_shape=(out_h, out_w)).astype(float) * _ANGLE_SCALE
                grid_t = src.transform * Affine.scale(src.width / out_w, src.height / out_h)
            if rows is None:
                # angle-grid sample centres expressed in pan-grid pixel coordinates
                cols = (grid_t.c + (numpy.arange(out_w) + 0.5) * grid_t.a - pan.c) / pan.a - 0.5
                rows = (grid_t.f + (numpy.arange(out_h) + 0.5) * grid_t.e - pan.f) / pan.e - 0.5
            fields[name] = self._field(rows, cols, values, name=name, units="degrees")

        want = ("SUN_ELEVATION", "SUN_AZIMUTH", "EARTH_SUN_DISTANCE", "ROLL_ANGLE")
        attrs = {
            k.lower(): float(self._image_attrs[k])
            for k in want
            if k in self._image_attrs
        }
        return Metadata(fields, attrs)

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
