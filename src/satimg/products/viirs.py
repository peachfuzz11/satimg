"""VIIRS Day/Night Band SDR reader (JPSS ``SVDNB`` + ``GDNBO`` HDF5)."""

from __future__ import annotations

import datetime
from functools import cached_property

import numpy
import PIL.Image
import xarray
from rasterio.control import GroundControlPoint

from satimg import viirs_utils
from satimg.metadata import Metadata
from satimg.product import Product
from satimg.readers import fit_thumbnail, keep_open, merge_bands, open_h5, vrt
from satimg.registry import register
from satimg.scene_geometry import SceneGeometry, coarsen
from satimg.source import Source
from satimg.tiling import as_band_yx, label_bands
from satimg.transform import SwathTransformer

#: metres the product's own transformer may be off the GEO file's per-pixel
#: positions (a DNB pixel is ~750 m)
TRANSFORMER_TOLERANCE_M = 50.0
#: ... and :attr:`ViirsDNBProduct.geometry`'s stored one: a third of a pixel,
#: at a third of the size
GEOMETRY_TOLERANCE_M = 250.0
#: side of the square grid the stored geometry keeps each angle field on
GEOMETRY_GRID = 32

#: nW / (cm^2 sr) per the SDR's W / (cm^2 sr)
NANOWATTS = 1e9

#: (low, high) radiance, nW / (cm^2 sr), the visual's log stretch spans:
#: night shows moonlit cloud to bright boat and city lights; day, sunlit ground
_NIGHT_STRETCH = (0.5, 500.0)
_DAY_STRETCH = (3e4, 3e7)


class ViirsGeometry(SceneGeometry):
    """A VIIRS DNB granule's :class:`~satimg.scene_geometry.SceneGeometry`:
    loads its transformer as a :class:`~satimg.transform.SwathTransformer`."""

    transformer_cls = SwathTransformer


@register(r"^SVDNB_(npp|j0\d)_d\d{8}_t\d{7}_e\d{7}_b\d+")
class ViirsDNBProduct(Product):
    """One VIIRS Day/Night Band granule: about 85 seconds of orbit, 768 x 4064
    pixels of ~750 m, in sensor (swath) geometry -- not map projected.

    Open the ``SVDNB_*.h5`` itself (its ``GDNBO_*.h5`` geolocation next to
    it), or a folder / zip named for it holding both, as the ``viirs-dnb``
    connector downloads.

    ``raw`` is the calibrated radiance, W / (cm^2 sr), ``float32`` (times 1e9
    for the nW / (cm^2 sr) night-lights work quotes), ``NaN`` where the SDR
    has a fill value. ``visual`` is a log stretch, one greyscale band,
    spanning moonlit cloud to boat lights at night. ``metadata`` carries the
    satellite / solar / lunar zenith and azimuth angles, and ``attrs`` the
    moon's illumination -- what decides how dark the background is.

    The transformer interpolates the GEO file's per-pixel positions to within
    :data:`TRANSFORMER_TOLERANCE_M`. Boats show up at night as their lights,
    a pixel or two each, not as hulls: see
    :func:`satimg.prediction.viirs.detect_boats` rather than the image-model
    :class:`~satimg.prediction.Detector`.
    """

    def __init__(self, source: Source):
        super().__init__(source)
        self._sdr, self._geo = viirs_utils.find_pair(self._source)
        with open_h5(self._source, self._sdr) as h5:
            granule = viirs_utils.read_granule(h5)
        with open_h5(self._source, self._geo) as h5:
            geolocation = viirs_utils.read_geolocation(h5)
        self._timestamp = granule["timestamp"]
        self._footprint = granule["footprint"]
        self._shape = geolocation["lat"].shape
        self._attrs = {
            **granule["attrs"],
            "moon_illumination": geolocation["moon_illumination"],
            "moon_phase_angle": geolocation["moon_phase_angle"],
        }
        fine = viirs_utils.nodes(geolocation, TRANSFORMER_TOLERANCE_M)
        coarse = viirs_utils.nodes(geolocation, GEOMETRY_TOLERANCE_M)
        self._transformer = SwathTransformer(
            fine["rows"], fine["cols"], fine["lat"], fine["lon"], scan_rows=viirs_utils.SCAN_ROWS
        )
        self._coarse_transformer = SwathTransformer(
            coarse["rows"], coarse["cols"], coarse["lat"], coarse["lon"], scan_rows=viirs_utils.SCAN_ROWS
        )
        self._angles = coarse

    # -- pixels ---------------------------------------------------------------
    @cached_property
    def _radiance_path(self) -> str:
        """The SDR's radiance through a VRT carrying a few GCPs -- just so
        GDAL sees it as georeferenced; satimg's own :attr:`transformer` is
        what maps it."""
        t = self._coarse_transformer
        rows = numpy.linspace(0, self.height, 5)
        cols = numpy.linspace(0, self.width, 5)
        rr, cc = (a.ravel() for a in numpy.meshgrid(rows, cols, indexing="ij"))
        latlon = t.rowcol_to_latlon(numpy.column_stack((rr, cc)))
        gcps = [
            GroundControlPoint(row=r, col=c, x=lon, y=lat)
            for r, c, (lat, lon) in zip(rr, cc, latlon)
        ]
        return vrt(viirs_utils.radiance_path(self._source.raster_path(self._sdr)), self._shape, gcps=gcps)

    def _open_raw(self, tile: int | tuple[int, int], overview: int = 1) -> xarray.DataArray:
        if overview != 1:
            raise ValueError("a VIIRS SDR has no overviews")
        merged = merge_bands([self._radiance_path], tile=tile)
        da = label_bands(as_band_yx(merged), ("DNB",))
        return keep_open(da.where(da > viirs_utils.FILL_BELOW), merged)

    def _render_visual(
        self, raw: xarray.DataArray, tile: int | tuple[int, int]
    ) -> xarray.DataArray:
        low, high = _DAY_STRETCH if self._attrs["day_night"] == "day" else _NIGHT_STRETCH
        log = numpy.log10((raw * NANOWATTS).clip(low, high)).isel(band=0)
        u8 = ((log - numpy.log10(low)) * (255 / numpy.log10(high / low))).fillna(0).astype("uint8")
        return label_bands(as_band_yx(u8), ("radiance",))

    def thumbnail(self) -> PIL.Image.Image:
        """Rendered from :attr:`visual` -- a whole granule is only ~12 MB;
        an SDR ships no quick-look of its own."""
        return fit_thumbnail(PIL.Image.fromarray(self.visual.values[0], mode="L"))

    # -- geometry -------------------------------------------------------------
    def _read_metadata(self) -> Metadata:
        a = self._angles
        fields = {
            name: self._field(a["rows"], a["cols"], a[dataset], name=name, units=units)
            for dataset, (name, units) in viirs_utils.ANGLE_FIELDS.items()
        }
        return Metadata(fields, self._attrs)

    @property
    def height(self) -> int:
        return self._shape[0]

    @property
    def width(self) -> int:
        return self._shape[1]

    @property
    def transformer(self) -> SwathTransformer:
        return self._transformer

    @property
    def geometry(self) -> ViirsGeometry:
        """As :attr:`Product.geometry`: the transformer at
        :data:`GEOMETRY_TOLERANCE_M`, the angle fields on a
        :data:`GEOMETRY_GRID` square grid -- a few hundred KB of JSON."""
        shape = (self.height, self.width)
        return ViirsGeometry(
            self._coarse_transformer, coarsen(self.metadata, shape, GEOMETRY_GRID), self._attrs, shape
        )

    @property
    def timestamp(self) -> datetime.datetime:
        return self._timestamp

    @property
    def footprint(self) -> dict:
        return self._footprint
