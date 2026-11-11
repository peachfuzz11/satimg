"""Capella GEC (geocoded, ellipsoid-corrected) reader."""

from __future__ import annotations

import numpy
import PIL.Image
import xarray

from satimg.metadata.metadata import Metadata
from satimg.product import Product
from satimg.products.capella.gec import capella_gec_parsing
from satimg.readers.raster import keep_open, merge_bands, raster_grid
from satimg.readers.thumbnail import load_thumbnail, thumbnail_from_visual
from satimg.registry import register
from satimg.sar_orbit_geometry import SAROrbitGeometry
from satimg.source.source import Source
from satimg.tiling import as_band_yx, label_bands
from satimg.transform.affine_transformer import AffineTransformer

#: where the visual's sigmoid stretch is centred / how steep it is (sigma0,
#: dB). Open sea in a Capella spotlight is at or near the noise floor -- its
#: NESZ is around -12 to -20 dB -- land around -10 to 0 dB, a ship well above
#: 0 dB: so sea comes out dark, land mid-grey and ships bright.
_VISUAL_CENTRE_DB = -4.0
_VISUAL_SLOPE = 0.3


@register(r"^CAPELLA_C\d{2}_(SP|SM|SS)_GEC_[HV]{2}_\d{14}_\d{14}(\.tif)?$")
class CapellaGEC(Product):
    """One Capella X-band SAR collect (spotlight, sliding spotlight or
    stripmap, ~0.5-1 m): its GEC image -- the least-processed geocoded
    product Capella ships: detected, multilooked and projected onto the
    WGS 84 ellipsoid (raised to the scene's mean height), not terrain
    corrected.

    Open the product folder (``CAPELLA_C13_SP_GEC_HH_..._20260909150146/``,
    its ``.tif`` + ``_extended.json``), a zip of it, or the ``.tif`` itself
    with the ``_extended.json`` beside it.

    ``raw`` is the GEC's one band, labelled by its polarisation: ``uint16``
    amplitude numbers, ``0`` outside the imaged area, on a north-up UTM grid.
    :meth:`sigma0` turns them into calibrated sigma-nought backscatter;
    ``visual`` is a dB sigmoid stretch of that, one greyscale band, as for
    Sentinel-1. ``metadata`` has no per-pixel fields; its ``attrs`` carry the
    scene-centre imaging geometry (``incidence_angle``, ``look_angle``,
    ``squint_angle``, ``look_direction``, ``pass``, ...) and the
    ``scale_factor``.

    The moving-target methods -- :meth:`doppler_azimuth_shift_m`,
    :meth:`doppler_pixel_shift`, :meth:`correct_position`,
    :meth:`heading_to_los` -- work from the sensor orbit in ``attrs["orbit"]``
    (the extended metadata's state vectors) and live on
    :class:`~satimg.sar_orbit_geometry.SAROrbitGeometry`.
    """

    def __init__(self, source: Source):
        super().__init__(source)
        tif, meta, self._thumb = capella_gec_parsing.find_files(self._source)
        self._tif = self._source.raster_path(tif)
        parsed = capella_gec_parsing.read_metadata(self._source, meta)
        self._timestamp = parsed["timestamp"]
        self._scale = parsed["scale_factor"]
        self._attrs = parsed["attrs"]
        transform, crs, self._shape = raster_grid(self._tif)
        self._transformer = AffineTransformer(transform, crs)
        self._footprint = self._grid_outline()
        self._geometry = SAROrbitGeometry(self._transformer, {}, self._attrs, self._shape)

    def _grid_outline(self) -> dict:
        """The image grid's outline as a GeoJSON Feature -- the extended
        metadata carries no footprint, and the grid's corners need no pixel
        read (the imaged area inside it is a little smaller: the grid is
        north-up, the collect is not)."""
        h, w = self._shape
        corners = self._transformer.rowcol_to_latlon([(0, 0), (0, w), (h, w), (h, 0), (0, 0)])
        ring = [(float(lon), float(lat)) for lat, lon in corners]
        return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]}}

    # -- pixels ----------------------------------------------------------------
    def sigma0(self, raw: xarray.DataArray) -> xarray.DataArray:
        """``raw`` (or any slice of it) as linear sigma-nought backscatter,
        ``(DN * scale_factor)^2``, ``float32``; ``NaN`` where ``raw`` is 0
        (outside the imaged area)."""
        amplitude = raw.where(raw > 0).astype("float32") * numpy.float32(self._scale)
        return amplitude**2

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
        """The quick-look PNG Capella publishes with the GEO product, if it
        was downloaded alongside; else rendered from :attr:`visual` at the
        GEC's smallest overview."""
        if self._thumb is not None:
            return load_thumbnail(self._source, self._thumb, greyscale=True)
        return thumbnail_from_visual(self.visual_overview(self.visual_overviews[-1]))

    # -- metadata ----------------------------------------------------------------
    def _read_metadata(self) -> Metadata:
        return self._geometry.metadata

    @property
    def height(self) -> int:
        """From the GeoTIFF header -- no ``raw`` open needed."""
        return self._shape[0]

    @property
    def width(self) -> int:
        return self._shape[1]

    # -- SAR geometry (see SAROrbitGeometry) ----------------------------------------
    @property
    def geometry(self) -> SAROrbitGeometry:
        """As :attr:`Product.geometry`, as a
        :class:`~satimg.sar_orbit_geometry.SAROrbitGeometry` -- which the
        moving-target methods below delegate to."""
        return self._geometry

    def heading_to_los(self, rowcol, heading_deg):
        """See :meth:`satimg.sar_orbit_geometry.SAROrbitGeometry.heading_to_los`."""
        return self._geometry.heading_to_los(rowcol, heading_deg)

    def doppler_azimuth_shift_m(self, rowcol, speed: float, heading_deg: float):
        """See :meth:`satimg.sar_orbit_geometry.SAROrbitGeometry.doppler_azimuth_shift_m`."""
        return self._geometry.doppler_azimuth_shift_m(rowcol, speed, heading_deg)

    def doppler_pixel_shift(self, rowcol, speed: float, heading_deg: float):
        """See :meth:`satimg.sar_orbit_geometry.SAROrbitGeometry.doppler_pixel_shift`."""
        return self._geometry.doppler_pixel_shift(rowcol, speed, heading_deg)

    def correct_position(self, lat: float, lon: float, speed: float, heading_deg: float):
        """See :meth:`satimg.sar_orbit_geometry.SAROrbitGeometry.correct_position`."""
        return self._geometry.correct_position(lat, lon, speed, heading_deg)
