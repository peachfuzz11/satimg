"""ICEYE GRD (ground range detected) reader."""

from __future__ import annotations

import numpy
import PIL.Image
import xarray

from satimg.metadata.metadata import Metadata
from satimg.products.iceye.grd import iceye_grd_parsing
from satimg.readers.raster import keep_open, merge_bands
from satimg.readers.thumbnail import thumbnail_from_visual
from satimg.registry import register
from satimg.sar_orbit_geometry import SAROrbitGeometry
from satimg.sar_product import SARProduct
from satimg.source.source import Source
from satimg.tiling import as_band_yx, label_bands
from satimg.transform.swath_transformer import SwathTransformer
from satimg.transform.transformer_utils import pixel_spacing

#: where the visual's sigmoid stretch is centred / how steep it is (sigma0,
#: dB) -- as for Capella: calm sea around -20 to -10 dB, wind-roughened sea
#: and land around -10 to 0 dB, ships well above 0 dB.
_VISUAL_CENTRE_DB = -4.0
_VISUAL_SLOPE = 0.3


@register(r"^ICEYE_[0-9A-Z]{6}_\d{8}T\d{6}Z_\d+_X\d+_[0-9A-Z]+(_GRD\.tif)?$")
class IceyeGRD(SARProduct):
    """One ICEYE X-band SAR image (spotlight ~0.5-1 m, stripmap ~3 m, scan
    ~15 m): its GRD -- the least-processed geocoded product ICEYE publishes:
    detected, multilooked and projected to ground range on the ellipsoid, in
    the radar's own geometry (not map-projected), located by a grid of
    ground control points.

    Open the product folder (``ICEYE_4EKGVZ_20251110T124428Z_7004635_X42_SLEDF/``
    with its ``_GRD.tif``), a zip of it, or the ``_GRD.tif`` itself -- the
    GeoTIFF carries all of its own metadata.

    ``raw`` is the GRD's one band, labelled by its polarisation: ``uint16``
    numbers packing the amplitude (GDAL band scale / offset), ``0`` outside
    the imaged area. :meth:`sigma0` unpacks and calibrates them; ``visual``
    is a dB sigmoid stretch of that, one greyscale band, as for Sentinel-1.
    The grid is sensor geometry, so ``transformer`` interpolates the GCP grid
    (a :class:`~satimg.transform.swath_transformer.SwathTransformer`), not an
    affine. ``metadata`` has no per-pixel fields; its ``attrs`` carry the
    scene-centre imaging geometry (``incidence_angle``, ``look_angle``,
    ``squint_angle``, ``look_direction``, ``pass``, ...) and the
    ``calibration_factor``.

    The moving-target correction and heading conversions
    (:class:`~satimg.sar_product.SARProduct`) work from the sensor orbit in ``attrs["orbit"]``
    (ICEYE's own orbit state vectors).
    """

    def __init__(self, source: Source):
        super().__init__(source)
        self._tif = self._source.raster_path(iceye_grd_parsing.find_grd(self._source))
        parsed = iceye_grd_parsing.read_metadata(self._tif)
        self._timestamp = parsed["timestamp"]
        self._cf = parsed["calibration_factor"]
        self._scale, self._offset = parsed["scale"], parsed["offset"]
        self._shape = parsed["shape"]
        self._attrs = parsed["attrs"]
        rows, cols, lat, lon = parsed["nodes"]
        self._transformer = SwathTransformer(rows + 0.5, cols + 0.5, lat, lon)
        self._attrs["pixel_spacing"] = list(pixel_spacing(self._transformer, self._shape))
        self._footprint = self._grid_outline()
        self._geometry = SAROrbitGeometry(self._transformer, {}, self._attrs, self._shape)

    def _grid_outline(self) -> dict:
        """The image grid's outline as a GeoJSON Feature, from the GCP grid
        -- a GRD's grid is the imaged area."""
        h, w = self._shape
        corners = self._transformer.rowcol_to_latlon([(0, 0), (0, w), (h, w), (h, 0), (0, 0)])
        ring = [(float(lon), float(lat)) for lat, lon in corners]
        return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]}}

    # -- pixels ----------------------------------------------------------------
    def sigma0(self, raw: xarray.DataArray) -> xarray.DataArray:
        """``raw`` (or any slice of it) as linear sigma-nought backscatter,
        ``calibration_factor * (DN * scale + offset)^2``, ``float32``;
        ``NaN`` where ``raw`` is 0 (outside the imaged area) or below the
        packing's floor (amplitude <= 0: no measurable signal)."""
        amplitude = raw.where(raw > 0).astype("float32") * numpy.float32(self._scale) + numpy.float32(
            self._offset
        )
        return numpy.float32(self._cf) * amplitude.where(amplitude > 0) ** 2

    def _open_raw(self, tile: int | tuple[int, int], overview: int = 1) -> xarray.DataArray:
        merged = merge_bands([self._tif], tile=tile, overview=overview)
        return keep_open(label_bands(as_band_yx(merged), self._attrs["polarisations"]), merged)

    def _visual_paths(self) -> list[str]:
        return [self._tif]

    def _render_visual(
        self, raw: xarray.DataArray, tile: int | tuple[int, int]
    ) -> xarray.DataArray:
        db = (10 * numpy.log10(self.sigma0(raw))).mean("band")
        u8 = 255 / (1 + numpy.exp(-(db - _VISUAL_CENTRE_DB) * _VISUAL_SLOPE))
        u8 = u8.fillna(0).clip(0, 255).astype("uint8")
        return label_bands(as_band_yx(u8), ("amplitude",))

    def thumbnail(self) -> PIL.Image.Image:
        """Rendered from :attr:`visual` at the GRD's smallest overview."""
        return thumbnail_from_visual(self.visual_overview(self.visual_overviews[-1]))

    # -- metadata ----------------------------------------------------------------
    def _read_metadata(self) -> Metadata:
        return self._geometry.metadata
