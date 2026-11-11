"""Umbra GEC (geocoded, ellipsoid-corrected) reader."""

from __future__ import annotations

import numpy
import PIL.Image
import xarray

from satimg.metadata.metadata import Metadata
from satimg.product import Product
from satimg.products.umbra.gec import umbra_gec_parsing
from satimg.readers.raster import keep_open, merge_bands, raster_grid
from satimg.readers.thumbnail import thumbnail_from_visual
from satimg.registry import register
from satimg.source.source import Source
from satimg.tiling import as_band_yx, label_bands
from satimg.transform.affine_transformer import AffineTransformer
from satimg.transform.transformer_utils import pixel_spacing

#: where the visual's sigmoid stretch is centred / how steep it is, in dB of
#: the GEC's own amplitude numbers (``20 log10 DN``; uncalibrated): calm
#: water is DN 1-6 (0-16 dB), land around DN 10-40, ships and steel up to
#: the 255 ceiling (48 dB).
_VISUAL_CENTRE_DB = 28.0
_VISUAL_SLOPE = 0.2


@register(r"^\d{4}(-\d{2}){5}_UMBRA-\d{2}(_GEC\.tif)?$")
class UmbraGEC(Product):
    """One Umbra X-band SAR collect (spotlight, ~0.25-1 m): its GEC image --
    the least-processed geocoded product Umbra ships: detected and projected
    onto the WGS 84 ellipsoid, not terrain corrected (at sea level that is
    exact).

    Open the collect folder (``2025-01-31-14-10-48_UMBRA-08/``, its
    ``_GEC.tif`` + metadata JSON), a zip of it, or the ``_GEC.tif`` itself
    with the metadata JSON beside it.

    ``raw`` is the GEC's one band, labelled by its polarisation: ``uint8``
    amplitude in Umbra's own scaling, not radiometrically calibrated, ``0``
    outside the imaged area. ``visual`` is a dB sigmoid stretch of it, one
    greyscale band, as for Sentinel-1: water dark, ships bright. The grid is EPSG:4326, rotated to the
    collect's own orientation (so ``transformer`` is a full affine, not
    north-up). ``metadata`` has no per-pixel fields; its ``attrs`` carry the
    scene-centre imaging geometry (``incidence_angle``, ``squint_angle``,
    ``look_direction``, ``pass``, ``slant_range``, ...).
    """

    def __init__(self, source: Source):
        super().__init__(source)
        tif, meta = umbra_gec_parsing.find_files(self._source)
        self._tif = self._source.raster_path(tif)
        parsed = umbra_gec_parsing.read_metadata(self._source, meta)
        self._timestamp = parsed["timestamp"]
        self._footprint = parsed["footprint"]
        self._attrs = parsed["attrs"]
        transform, crs, self._shape = raster_grid(self._tif)
        self._transformer = AffineTransformer(transform, crs)
        self._attrs["pixel_spacing"] = list(pixel_spacing(self._transformer, self._shape))

    def _open_raw(self, tile: int | tuple[int, int], overview: int = 1) -> xarray.DataArray:
        merged = merge_bands([self._tif], tile=tile, overview=overview)
        names = self._attrs["polarisations"][:1] or ["amplitude"]
        return keep_open(label_bands(as_band_yx(merged), names), merged)

    def _visual_paths(self) -> list[str]:
        return [self._tif]

    def _render_visual(
        self, raw: xarray.DataArray, tile: int | tuple[int, int]
    ) -> xarray.DataArray:
        db = (20 * numpy.log10(raw.where(raw > 0).astype("float32"))).mean("band")
        u8 = 255 / (1 + numpy.exp(-(db - _VISUAL_CENTRE_DB) * _VISUAL_SLOPE))
        u8 = u8.fillna(0).clip(0, 255).astype("uint8")
        return label_bands(as_band_yx(u8), ("amplitude",))

    def thumbnail(self) -> PIL.Image.Image:
        """Rendered from the GEC's smallest overviews (Umbra ships no
        quick-look)."""
        return thumbnail_from_visual(self.visual_overview(self.visual_overviews[-1]))

    def _read_metadata(self) -> Metadata:
        return Metadata({}, self._attrs)

    @property
    def height(self) -> int:
        """From the GeoTIFF header -- no ``raw`` open needed."""
        return self._shape[0]

    @property
    def width(self) -> int:
        return self._shape[1]
