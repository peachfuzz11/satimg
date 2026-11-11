"""NISAR L2 GCOV (geocoded polarimetric covariance) reader."""

from __future__ import annotations

from functools import cached_property

import numpy
import PIL.Image
import rasterio
import xarray

from satimg.metadata.metadata import Metadata
from satimg.products.nisar.gcov import nisar_gcov_parsing
from satimg.readers.hdf5 import open_h5
from satimg.readers.raster import keep_open, merge_bands
from satimg.readers.thumbnail import THUMBNAIL_SIZE, load_thumbnail
from satimg.readers.vrt import vrt
from satimg.registry import register
from satimg.sar_orbit_geometry import SAROrbitGeometry
from satimg.sar_product import SARProduct
from satimg.scene_geometry import coarsen
from satimg.source.source import Source
from satimg.tiling import as_band_yx, label_bands
from satimg.transform.affine_transformer import AffineTransformer

#: side of the square grid :attr:`NisarGCOV.geometry` stores each
#: radar-grid field on
GEOMETRY_GRID = 32

#: where the visual's sigmoid stretch is centred / how steep it is (dB): L-band
#: open sea sits around -25 dB, a ship well above 0 dB.
_VISUAL_CENTRE_DB = -15.0
_VISUAL_SLOPE = 0.25


@register(r"^NISAR_L2_[A-Z]{2}_GCOV_\d{3}_\d{3}_[AD]_\d{3}_.*?(\.h5)?$")
class NisarGCOV(SARProduct):
    """A NISAR L-band GCOV granule -- one ``.h5`` file, opened directly (or
    from a zip holding it).

    ``raw`` holds the diagonal covariance terms of frequency A (``HHHH``,
    ``HVHV``, ...: radiometrically terrain-corrected gamma-0 backscatter
    *power*, linear, ``float32``, ``NaN`` outside the swath) on the product's
    own north-up map grid (UTM, or polar stereographic at high latitude);
    ``visual`` is a dB sigmoid stretch, one greyscale band, as for
    Sentinel-1. Off-diagonal (complex) terms and the 80 m frequency-B grid
    are not read.

    ``metadata`` carries the radar-grid cube sampled at sea level
    (``incidence_angle``, ``elevation_angle``, ``slant_range``, the
    ``los_*`` / ``along_track_*`` unit vectors, ``ground_track_velocity``).
    ``attrs["orbit"]`` holds the product's orbit state vectors, which the
    moving-target correction and heading conversions
    (:class:`~satimg.sar_product.SARProduct`) work from.
    """

    def __init__(self, source: Source):
        super().__init__(source)
        self._h5 = nisar_gcov_parsing.find_h5(self._source)
        with open_h5(self._source, self._h5) as h5:
            ident = nisar_gcov_parsing.read_identification(h5)
            grid = nisar_gcov_parsing.read_grid(h5)
            cube = nisar_gcov_parsing.read_radar_grid(h5, grid["transform"])
            orbit = nisar_gcov_parsing.orbit(h5, ident["timestamp"])
        self._timestamp = ident["timestamp"]
        self._footprint = ident["footprint"]
        self._terms = grid["terms"]
        attrs = {
            **ident["attrs"],
            "polarisations": list(self._terms),
            "pixel_spacing": list(grid["pixel_spacing"]),
            "orbit": orbit,
        }
        fields = {
            name: (cube["rows"], cube["cols"], cube[name], units)
            for name, units in {**nisar_gcov_parsing.CUBE_FIELDS, **nisar_gcov_parsing.PLANE_FIELDS}.values()
            if name in cube
        }
        self._geometry = SAROrbitGeometry(
            AffineTransformer(grid["transform"], grid["crs"]), fields, attrs, grid["shape"]
        )

    # -- pixels ----------------------------------------------------------------
    @cached_property
    def _term_paths(self) -> list[str]:
        t = self.transformer
        return [
            vrt(
                nisar_gcov_parsing.gdal_path(self._source, self._h5, f"{nisar_gcov_parsing.GRID}/{term}"),
                (self.height, self.width), transform=t.transform, crs=t.crs,
            )
            for term in self._terms
        ]

    def _open_raw(self, tile: int | tuple[int, int], overview: int = 1) -> xarray.DataArray:
        merged = merge_bands(self._term_paths, tile=tile, overview=overview)
        return keep_open(label_bands(as_band_yx(merged), self._terms), merged)

    def _render_visual(
        self, raw: xarray.DataArray, tile: int | tuple[int, int]
    ) -> xarray.DataArray:
        db = (10 * numpy.log10(raw.where(raw > 0))).mean("band")
        u8 = 255 / (1 + numpy.exp(-(db - _VISUAL_CENTRE_DB) * _VISUAL_SLOPE))
        u8 = u8.fillna(0).clip(0, 255).astype("uint8")
        return label_bands(as_band_yx(u8), ("amplitude",))

    def thumbnail(self) -> PIL.Image.Image:
        """A browse PNG shipped next to the ``.h5`` (as ASF serves them,
        ``<granule>*.png``) if there is one, else rendered from the first
        covariance term read decimated -- which still decodes every chunk of
        it, so a few seconds on a full frame."""
        stem = self._h5[: -len(".h5")]
        browse = sorted(n for n in self._source.listdir() if n.startswith(stem) and n.endswith(".png"))
        if browse:
            return load_thumbnail(self._source, browse[0], greyscale=True)
        scale = max(self.height, self.width) / max(THUMBNAIL_SIZE)
        shape = (max(1, round(self.height / scale)), max(1, round(self.width / scale)))
        with rasterio.open(self._term_paths[0]) as src:
            power = src.read(1, out_shape=shape)
        raw = xarray.DataArray(power[numpy.newaxis], dims=("band", "y", "x"))
        img = PIL.Image.fromarray(self._render_visual(raw, shape).values[0], mode="L")
        return img.resize(THUMBNAIL_SIZE)

    # -- geometry ----------------------------------------------------------------
    def _read_metadata(self) -> Metadata:
        return self._geometry.metadata

    @property
    def geometry(self) -> SAROrbitGeometry:
        """As :attr:`Product.geometry`, with the radar-grid fields resampled
        to a :data:`GEOMETRY_GRID` square grid to keep it small to store --
        they vary smoothly across the scene."""
        shape = (self.height, self.width)
        return SAROrbitGeometry(
            self.transformer, coarsen(self.metadata, shape, GEOMETRY_GRID), self.metadata.attrs, shape
        )
