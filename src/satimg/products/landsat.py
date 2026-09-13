"""Landsat Collection-2 Level-1 reader."""

from __future__ import annotations

import datetime
import os

import numpy
import xarray

from satimg import landsat_utils
from satimg.landsat_utils import BANDS
from satimg.metadata import Metadata
from satimg.product import Product
from satimg.readers import keep_open, load_thumbnail, merge_bands
from satimg.registry import register
from satimg.source import Source
from satimg.tiling import as_band_yx, label_bands
from satimg.transform import Transformer


@register(r"^LC(0[1-9])_L1(TP|GT)_\d+_\d+_\d+_\d+_T1$")
class LandsatProduct(Product):
    """``raw`` is every band resampled to the 15 m panchromatic grid; ``visual`` is
    a 3-band pan-sharpened true-colour render. ``metadata`` carries the per-pixel
    angle rasters (``sun_zenith``, ``sun_azimuth``, ``view_zenith``,
    ``view_azimuth``), read decimated to a coarse grid. ``MTL.json``/angle-raster
    parsing itself lives in :mod:`satimg.landsat_utils`."""

    def __init__(self, source: Source):
        super().__init__(source)
        mtl = landsat_utils.parse_mtl(self._source)
        self._image_attrs = mtl["image_attrs"]
        self._timestamp = mtl["timestamp"]
        self._footprint = mtl["footprint"]
        self._transformer = mtl["transformer"]

    def _open_raw(self, tile: int | tuple[int, int]) -> xarray.DataArray:
        self._require_extracted("raw")
        paths = [os.path.join(self._path, f"{b}.TIF") for b in BANDS]
        merged = merge_bands(paths, match=landsat_utils.PAN, tile=tile)
        return keep_open(label_bands(as_band_yx(merged), BANDS), merged)

    def _render_visual(
        self, raw: xarray.DataArray, tile: int | tuple[int, int]
    ) -> xarray.DataArray:
        self._require_extracted("visual")
        a = raw.drop_vars("band")  # positional indexing below; relabel at the end
        rgb = a.isel(band=list(landsat_utils.RGB)).astype("float32")
        weights = numpy.array([0.3, 0.3, 0.35], dtype="float32")
        rgb = rgb * weights[:, None, None]
        rgb = rgb / (rgb.sum(dim="band") + 1e-12)
        rgb = rgb * a.isel(band=landsat_utils.PAN).astype("float32")
        rgb = 0.75 * rgb.fillna(0) ** (1 / 1.4)
        return label_bands(as_band_yx(rgb.clip(0, 255).astype("uint8")),
                           ("red", "green", "blue"))

    def _read_metadata(self) -> Metadata:
        # no XML/JSON equivalent ships for these -- only ever as full rasters.
        self._require_extracted("metadata")
        pan_transform = self._transformer._transform
        fields = {}
        for name, fname in landsat_utils.ANGLE_FILES.items():
            path = os.path.join(self._path, fname)
            rows, cols, values = landsat_utils.read_angle_grid(path, pan_transform)
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
        entries = self._source.listdir()
        for suffix in ("_small.jpeg", "thumbnail.jpeg", ".jpeg"):
            match = next((o for o in entries if o.endswith(suffix)), None)
            if match:
                return load_thumbnail(self._source, match)
        raise FileNotFoundError(f"no thumbnail under {self._source.name}")
