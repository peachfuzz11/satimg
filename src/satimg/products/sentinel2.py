"""Sentinel-2 Level-1C reader."""

from __future__ import annotations

import datetime

import rioxarray
import xarray

from satimg import s2_utils
from satimg.metadata import Metadata
from satimg.product import Product
from satimg.readers import keep_open, load_thumbnail, merge_bands
from satimg.registry import register
from satimg.source import Source
from satimg.tiling import as_band_yx, label_bands
from satimg.transform import Transformer


@register(r"^S2[ABCD]_MSIL1C_\d{8}T\d{6}_.*_\d{8}T\d{6}\.SAFE$")
class Sentinel2L1CProduct(Product):
    """``raw`` is the 13 reflectance bands resampled to the 10 m grid; ``visual``
    is the shipped 3-band True Colour Image. ``metadata`` carries the ``MTD_TL.xml``
    angle grids (``sun_zenith``, ``sun_azimuth``, ``view_zenith``, ``view_azimuth``);
    the viewing grids are the per-detector grids merged. SAFE parsing itself
    lives in :mod:`satimg.s2_utils`."""

    def __init__(self, source: Source):
        super().__init__(source)
        self._meta = s2_utils.parse_mtd(self._source, self._path)
        self._tl = s2_utils.parse_tl(self._source)
        self._timestamp = self._tl["timestamp"]
        self._gsd = s2_utils.BAND_GSD_M[self._meta["band_names"][1]]
        self._transformer = Transformer(self._tl["transforms"][self._gsd], self._tl["crs"])

    def _open_raw(self, tile: int | tuple[int, int]) -> xarray.DataArray:
        self._require_extracted("raw")
        merged = merge_bands(self._meta["reflectance"], match=1, tile=tile)
        da = label_bands(as_band_yx(merged), self._meta["band_names"])
        da = da.assign_coords(wavelength_nm=("band", self._meta["wavelengths"]))
        return keep_open(da, merged)

    def _render_visual(
        self, raw: xarray.DataArray, tile: int | tuple[int, int]
    ) -> xarray.DataArray:
        self._require_extracted("visual")
        src = rioxarray.open_rasterio(self._meta["tci"])
        tw, th = (tile, tile) if isinstance(tile, int) else tile
        # a plain (unchunked) DataArray's .astype() computes eagerly -- chunk
        # first so this stays lazy and windowed like every other view.
        tci = as_band_yx(src.chunk({"x": tw, "y": th}).astype("uint8"))
        return keep_open(label_bands(tci, ("red", "green", "blue")), src)

    def _read_metadata(self) -> Metadata:
        rows, cols = self._tl["axes"]
        fields = {
            name: self._field(rows, cols, grid, name=name, units="degrees")
            for name, grid in self._tl["grids"].items()
        }
        return Metadata(fields, self._tl["attrs"])

    @property
    def height(self) -> int:
        """From ``MTD_TL.xml``'s ``Tile_Geocoding/Size`` at ``band_names[1]``'s
        resolution (matches ``raw`` exactly) rather than the base
        ``int(self.raw.sizes["y"])`` -- so ``metadata``'s fields get a real
        ``.grid`` / ``.corners()`` even zip-native, with no ``raw`` open
        needed."""
        return self._tl["pixel_shapes"][self._gsd][0]

    @property
    def width(self) -> int:
        """See :attr:`height`."""
        return self._tl["pixel_shapes"][self._gsd][1]

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
        name = next(o for o in self._source.listdir() if o.endswith("-ql.jpg"))
        return load_thumbnail(self._source, name)
