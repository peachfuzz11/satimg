"""VIIRS Day/Night Band L1B reader (NASA ``V??02DNB`` + ``V??03DNB``, netCDF4)."""

from __future__ import annotations

import datetime
from functools import cached_property

import numpy
import PIL.Image
import xarray
from rasterio.control import GroundControlPoint

from satimg.metadata.collection import Metadata
from satimg.product import Product
from satimg.products.viirs import files
from satimg.readers.hdf5 import open_h5
from satimg.readers.raster import keep_open, merge_bands
from satimg.readers.thumbnail import fit_thumbnail
from satimg.readers.vrt import vrt
from satimg.registry import register
from satimg.scene_geometry import SceneGeometry, coarsen
from satimg.source.base import Source
from satimg.tiling import as_band_yx, label_bands
from satimg.transform.swath import SwathTransformer

#: metres the product's own transformer may be off the GEO file's per-pixel
#: positions (a DNB pixel is ~750 m)
TRANSFORMER_TOLERANCE_M = 50.0
#: ... and :attr:`ViirsDNBProduct.geometry`'s stored one: a third of a pixel,
#: at a third of the size
GEOMETRY_TOLERANCE_M = 250.0
#: side of the square grid the stored geometry keeps each angle field on
GEOMETRY_GRID = 32

#: nW / (cm^2 sr) per the L1B's W / (cm^2 sr)
NANOWATTS = 1e9

#: (low, high) radiance, nW / (cm^2 sr), the visual's log stretch spans:
#: night shows moonlit cloud to bright boat and city lights; day, sunlit ground
_NIGHT_STRETCH = (0.5, 500.0)
_DAY_STRETCH = (3e4, 3e7)



@register(r"^V(NP|J\d)02DNB(_NRT)?\.A\d{7}\.\d{4}\.\d{3}")
class ViirsDNBProduct(Product):
    """One VIIRS Day/Night Band L1B granule -- NASA's radiance file with its
    geolocation file: NOAA-20's ``VJ102DNB_NRT`` + ``VJ103DNB_NRT`` or
    NOAA-21's ``VJ202DNB_NRT`` + ``VJ203DNB_NRT`` (near-real-time), or the
    same pairs from the standard archive (no ``_NRT``; SNPP's ``VNP02DNB`` /
    ``VNP03DNB`` share the format). 6 minutes of orbit, ~3232 x 4064 pixels
    of ~750 m, in sensor (swath) geometry -- not map projected.

    Open the radiance ``.nc`` itself (its geolocation file next to it), or a
    folder / zip named for it holding both, as the ``viirs-dnb`` connector
    downloads.

    ``raw`` is the calibrated radiance, W / (cm^2 sr), ``float32`` (times 1e9
    for the nW / (cm^2 sr) night-lights work quotes), ``NaN`` where the file
    has a fill value. ``visual`` is a log stretch, one greyscale band,
    spanning moonlit cloud to boat lights at night. ``metadata`` carries the
    satellite / solar / lunar zenith and azimuth angles, and ``attrs`` the
    moon's illumination -- what decides how dark the background is.

    The transformer interpolates the geolocation file's per-pixel positions
    to within :data:`TRANSFORMER_TOLERANCE_M`. Boats show up at night as
    their lights, a pixel or two each, not as hulls: see
    :class:`~satimg.prediction.ViirsBoatDetector` rather than the image-model
    :class:`~satimg.prediction.Detector`.
    """

    def __init__(self, source: Source):
        super().__init__(source)
        self._radiance_file, geo = files.find_pair(self._source)
        with open_h5(self._source, self._radiance_file) as h5:
            granule = files.read_granule(h5, self._radiance_file)
        with open_h5(self._source, geo) as h5:
            geolocation = files.read_geolocation(h5)
        self._timestamp = granule["timestamp"]
        self._footprint = granule["footprint"]
        self._shape = geolocation["lat"].shape
        self._attrs = {
            **granule["attrs"],
            "moon_illumination": geolocation["moon_illumination"],
            "moon_phase_angle": geolocation["moon_phase_angle"],
        }
        fine = files.nodes(geolocation, TRANSFORMER_TOLERANCE_M)
        coarse = files.nodes(geolocation, GEOMETRY_TOLERANCE_M)
        self._transformer = SwathTransformer(
            fine["rows"], fine["cols"], fine["lat"], fine["lon"], scan_rows=files.SCAN_ROWS
        )
        self._coarse_transformer = SwathTransformer(
            coarse["rows"], coarse["cols"], coarse["lat"], coarse["lon"], scan_rows=files.SCAN_ROWS
        )
        self._angles = coarse

    # -- pixels ---------------------------------------------------------------
    @cached_property
    def _radiance_path(self) -> str:
        """The radiance through a VRT carrying a few GCPs -- just so
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
        path = files.gdal_path(self._source.raster_path(self._radiance_file), files.RADIANCE)
        return vrt(path, self._shape, gcps=gcps)

    def _open_raw(self, tile: int | tuple[int, int], overview: int = 1) -> xarray.DataArray:
        if overview != 1:
            raise ValueError("a VIIRS granule has no overviews")
        merged = merge_bands([self._radiance_path], tile=tile)
        da = label_bands(as_band_yx(merged), ("DNB",))
        return keep_open(da.where(da > files.FILL_BELOW), merged)

    def _render_visual(
        self, raw: xarray.DataArray, tile: int | tuple[int, int]
    ) -> xarray.DataArray:
        low, high = _DAY_STRETCH if self._attrs["day_night"] == "day" else _NIGHT_STRETCH
        log = numpy.log10((raw * NANOWATTS).clip(low, high)).isel(band=0)
        u8 = ((log - numpy.log10(low)) * (255 / numpy.log10(high / low))).fillna(0).astype("uint8")
        return label_bands(as_band_yx(u8), ("radiance",))

    def thumbnail(self) -> PIL.Image.Image:
        """Rendered from :attr:`visual` -- neither format ships a quick-look."""
        return fit_thumbnail(PIL.Image.fromarray(self.visual.values[0], mode="L"))

    # -- geometry -------------------------------------------------------------
    def _read_metadata(self) -> Metadata:
        a = self._angles
        fields = {
            name: self._field(a["rows"], a["cols"], a[name], name=name, units="degrees")
            for name in files.FIELDS
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
    def geometry(self) -> SceneGeometry:
        """As :attr:`Product.geometry`: the transformer at
        :data:`GEOMETRY_TOLERANCE_M`, the angle fields on a
        :data:`GEOMETRY_GRID` square grid -- a few hundred KB of JSON."""
        shape = (self.height, self.width)
        return SceneGeometry(
            self._coarse_transformer, coarsen(self.metadata, shape, GEOMETRY_GRID), self._attrs, shape
        )

    @property
    def timestamp(self) -> datetime.datetime:
        return self._timestamp

    @property
    def footprint(self) -> dict:
        return self._footprint
