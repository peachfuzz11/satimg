"""VIIRS Day/Night Band L1B reader (NASA ``V??02DNB`` + ``V??03DNB``, netCDF4)."""

from __future__ import annotations

from functools import cached_property

import numpy
import PIL.Image
import xarray

from satimg.metadata.metadata import Metadata
from satimg.product import Product
from satimg.products.viirs.dnb import viirs_dnb_parsing
from satimg.readers.hdf5 import open_h5
from satimg.readers.raster import keep_open, merge_bands
from satimg.readers.thumbnail import thumbnail_from_visual
from satimg.readers.vrt import gcps, vrt
from satimg.registry import register
from satimg.scene_geometry import SceneGeometry, coarsen
from satimg.source.source import Source
from satimg.tiling import as_band_yx, label_bands
from satimg.transform.swath_transformer import SwathTransformer

#: metres the product's own transformer may be off the GEO file's per-pixel
#: positions (a DNB pixel is ~750 m)
TRANSFORMER_TOLERANCE_M = 50.0
#: ... and :attr:`ViirsDNB.geometry`'s stored one, on a plain grid that
#: interpolates across scans (see :func:`viirs_dnb_parsing.coarse_nodes`):
#: within ~1.3 km (under two pixels), in a few hundred nodes
GEOMETRY_TOLERANCE_M = 500.0
#: side of the square grid the stored geometry keeps each angle field on
GEOMETRY_GRID = 16

#: nW / (cm^2 sr) per the L1B's W / (cm^2 sr)
NANOWATTS = 1e9

#: (low, high) radiance, nW / (cm^2 sr), the visual's log stretch spans:
#: night shows the moonless sea (~0.1 - 0.5) to boat lights (~5 - 100), city
#: centres saturating; day, sunlit ground
_NIGHT_STRETCH = (0.1, 100.0)
_DAY_STRETCH = (3e4, 3e7)


@register(r"^V(NP|J\d)02DNB(_NRT)?\.A\d{7}\.\d{4}\.\d{3}")
class ViirsDNB(Product):
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
    :class:`~satimg.prediction.viirs_boat_detector.ViirsBoatDetector` rather than the image-model
    :class:`~satimg.prediction.detector.Detector`.
    """

    def __init__(self, source: Source):
        super().__init__(source)
        self._radiance_file, geo = viirs_dnb_parsing.find_pair(self._source)
        with open_h5(self._source, self._radiance_file) as h5:
            granule = viirs_dnb_parsing.read_granule(h5, self._radiance_file)
        with open_h5(self._source, geo) as h5:
            geolocation = viirs_dnb_parsing.read_geolocation(h5)
            fine = viirs_dnb_parsing.nodes(geolocation, TRANSFORMER_TOLERANCE_M)
            coarse = viirs_dnb_parsing.coarse_nodes(geolocation, GEOMETRY_TOLERANCE_M)
            coarse.update(viirs_dnb_parsing.read_angles(h5, coarse))
        self._timestamp = granule["timestamp"]
        self._footprint = granule["footprint"]
        self._shape = geolocation["shape"]
        self._attrs = {
            **granule["attrs"],
            "moon_illumination": geolocation["moon_illumination"],
            "moon_phase_angle": geolocation["moon_phase_angle"],
        }
        self._transformer = SwathTransformer(
            fine["rows"], fine["cols"], fine["lat"], fine["lon"], scan_rows=viirs_dnb_parsing.SCAN_ROWS
        )
        self._coarse_transformer = SwathTransformer(coarse["rows"], coarse["cols"], coarse["lat"], coarse["lon"])
        self._angles = coarse

    # -- pixels ---------------------------------------------------------------
    @cached_property
    def _radiance_path(self) -> str:
        """The radiance through a VRT carrying a few GCPs -- just so
        GDAL sees it as georeferenced; satimg's own :attr:`transformer` is
        what maps it."""
        path = viirs_dnb_parsing.gdal_path(self._source.raster_path(self._radiance_file), viirs_dnb_parsing.RADIANCE)
        return vrt(path, self._shape, gcps=gcps(self._coarse_transformer, self._shape))

    def _open_raw(self, tile: int | tuple[int, int], overview: int = 1) -> xarray.DataArray:
        if overview != 1:
            raise ValueError("a VIIRS granule has no overviews")
        merged = merge_bands([self._radiance_path], tile=tile)
        da = label_bands(as_band_yx(merged), ("DNB",))
        return keep_open(da.where(da > viirs_dnb_parsing.FILL_BELOW), merged)

    def _render_visual(
        self, raw: xarray.DataArray, tile: int | tuple[int, int]
    ) -> xarray.DataArray:
        low, high = _DAY_STRETCH if self._attrs["day_night"] == "day" else _NIGHT_STRETCH
        log = numpy.log10((raw * NANOWATTS).clip(low, high)).isel(band=0)
        u8 = ((log - numpy.log10(low)) * (255 / numpy.log10(high / low))).fillna(0).astype("uint8")
        return label_bands(as_band_yx(u8), ("radiance",))

    def thumbnail(self) -> PIL.Image.Image:
        """Rendered from :attr:`visual`, decimated -- neither format ships a
        quick-look."""
        return thumbnail_from_visual(self.visual)

    # -- geometry -------------------------------------------------------------
    def _read_metadata(self) -> Metadata:
        a = self._angles
        fields = {
            name: self._field(a["rows"], a["cols"], a[name], name=name, units="degrees")
            for name in viirs_dnb_parsing.FIELDS
        }
        return Metadata(fields, self._attrs)

    @property
    def height(self) -> int:
        return self._shape[0]

    @property
    def width(self) -> int:
        return self._shape[1]

    @property
    def geometry(self) -> SceneGeometry:
        """As :attr:`Product.geometry`: the transformer at
        :data:`GEOMETRY_TOLERANCE_M`, the angle fields on a
        :data:`GEOMETRY_GRID` square grid to the file's 0.01 degree -- tens of
        KB of JSON."""
        shape = (self.height, self.width)
        fields = {
            name: (rows, cols, numpy.round(values, 2), units)
            for name, (rows, cols, values, units) in coarsen(self.metadata, shape, GEOMETRY_GRID).items()
        }
        return SceneGeometry(self._coarse_transformer, fields, self._attrs, shape)
