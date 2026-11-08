"""Sentinel-3 OLCI Level-1B full resolution (``OL_1_EFR___``) reader."""

from __future__ import annotations

import datetime
from functools import cached_property

import numpy
import PIL.Image
import xarray
from rasterio.control import GroundControlPoint

from satimg.metadata.collection import Metadata
from satimg.product import Product
from satimg.products import swath
from satimg.products.sentinel3 import files
from satimg.readers.hdf5 import open_h5
from satimg.readers.raster import GDALRaster, Tile
from satimg.readers.stack import BandStack
from satimg.readers.thumbnail import load_thumbnail
from satimg.readers.vrt import vrt
from satimg.registry import register
from satimg.scene_geometry import SceneGeometry, coarsen
from satimg.source.base import Source
from satimg.tiling import as_band_yx, label_bands
from satimg.transform.swath import SwathTransformer

#: metres the transformer may be off the tie-point geolocation between its
#: node rows (pixels are ~300 m)
TRANSFORMER_TOLERANCE_M = 10.0
#: ... and :attr:`Sentinel3OLCIProduct.geometry`'s stored one
GEOMETRY_TOLERANCE_M = 100.0
#: side of the square grid the stored geometry keeps each angle field on
GEOMETRY_GRID = 32

#: the visual's true colour bands (R, G, B) and the reflectance its stretch tops out at
_RGB = ("Oa08", "Oa06", "Oa04")
_VISUAL_MAX = 0.3



@register(r"^S3[A-D]_OL_1_EFR____\d{8}T\d{6}_\d{8}T\d{6}_")
class Sentinel3OLCIProduct(Product):
    """A Sentinel-3 OLCI full-resolution Level-1B product (``.SEN3`` folder,
    or the zip Copernicus distributes it as): 21 bands, ~300 m pixels, a
    ~1270 km push-broom swath in sensor geometry -- ~3 minutes of orbit,
    ~4090 x 4865 pixels.

    ``raw`` is the 21 radiance bands as stored (``uint16``, ``Oa01`` ..
    ``Oa21``, ``wavelength_nm`` on the band axis, 65535 fill);
    ``metadata.attrs["radiance_scales"]`` turns them into mW / (m^2 sr nm),
    and :meth:`radiance` does it for you. ``visual`` is true colour from
    Oa08 / Oa06 / Oa04. ``metadata`` carries the sun / view angles from the
    tie-point grid.

    The transformer interpolates the tie-point geolocation (see
    :mod:`satimg.products.sentinel3.files`): smooth, within about a third of a pixel of
    the per-pixel positions at sea, but a mountain top lies off its
    terrain-corrected position by up to a few hundred metres. Daytime only; at 300 m only large ships,
    and wakes in sun glint, are visible, and the image models were not
    trained on OLCI.
    """

    def __init__(self, source: Source):
        super().__init__(source)
        manifest = self._source.find_file("xfdumanifest.xml")
        if manifest is None:
            raise FileNotFoundError(f"no xfdumanifest.xml under {self._source.name}")
        self._dir = manifest[: -len("xfdumanifest.xml")]
        with open_h5(self._source, f"{self._dir}tie_geo_coordinates.nc") as h5:
            start, stop = files.read_times(h5)
            geolocation = files.read_geolocation(h5)
        with open_h5(self._source, f"{self._dir}{files.band_file('Oa01')}") as h5:
            self._shape = files.read_shape(h5)
        with open_h5(self._source, f"{self._dir}tie_geometries.nc") as h5:
            self._angles = files.read_angles(h5)
        scales, offsets = files.read_scales(self._source, self._dir)
        self._timestamp = start + (stop - start) / 2
        self._footprint = files.read_footprint(self._source) or swath.outline(geolocation["lat"], geolocation["lon"])
        self._attrs = {
            **files.read_name(self._source.name),
            "radiance_scales": scales,
            "radiance_offsets": offsets,
        }

        def transformer(tolerance):
            n = files.nodes(geolocation, tolerance)
            return SwathTransformer(n["rows"], n["cols"], n["lat"], n["lon"])

        self._transformer = transformer(TRANSFORMER_TOLERANCE_M)
        self._coarse_transformer = transformer(GEOMETRY_TOLERANCE_M)

    # -- pixels ---------------------------------------------------------------
    @cached_property
    def _bands(self) -> BandStack:
        """Each band through a VRT carrying a few GCPs -- just so GDAL sees
        it as georeferenced; :attr:`transformer` is what maps it."""
        rows = numpy.linspace(0, self.height, 5)
        cols = numpy.linspace(0, self.width, 5)
        rr, cc = (a.ravel() for a in numpy.meshgrid(rows, cols, indexing="ij"))
        latlon = self._coarse_transformer.rowcol_to_latlon(numpy.column_stack((rr, cc)))
        gcps = [GroundControlPoint(row=r, col=c, x=lon, y=lat) for r, c, (lat, lon) in zip(rr, cc, latlon)]
        rasters = [
            GDALRaster(vrt(files.gdal_path(self._source.raster_path(f"{self._dir}{files.band_file(b)}"), b),
                           self._shape, gcps=gcps, dtype="UInt16", nodata=65535))
            for b in files.BANDS
        ]
        return BandStack(rasters, files.BANDS, files.WAVELENGTHS_NM)

    def _raw_bands(self) -> BandStack:
        return self._bands

    def radiance(self, raw: xarray.DataArray) -> xarray.DataArray:
        """``raw`` (or a band / window of it) in mW / (m^2 sr nm),
        ``float32``, fills ``NaN``."""
        pick = [files.BANDS.index(b) for b in raw.band.values]
        scales = numpy.array(self._attrs["radiance_scales"], dtype="float32")[pick, None, None]
        offsets = numpy.array(self._attrs["radiance_offsets"], dtype="float32")[pick, None, None]
        return raw.where(raw != 65535).astype("float32") * scales + offsets

    def _render_visual(self, raw: xarray.DataArray, tile: Tile) -> xarray.DataArray:
        rgb = self.radiance(raw.sel(band=list(_RGB)))
        e0 = numpy.array([files.SOLAR_IRRADIANCE[b] for b in _RGB], dtype="float32")[:, None, None]
        reflectance = numpy.pi * rgb / e0  # x cos(sun zenith), left out: fine for a picture
        u8 = (255 * (reflectance.clip(0, _VISUAL_MAX) / _VISUAL_MAX) ** (1 / 2.2)).fillna(0).astype("uint8")
        return label_bands(as_band_yx(u8.drop_vars("wavelength_nm", errors="ignore")), ("red", "green", "blue"))

    def thumbnail(self) -> PIL.Image.Image:
        """The product's shipped ``quicklook.jpg``, else rendered from
        :attr:`visual` (see :meth:`Product.thumbnail`)."""
        quicklook = self._source.find_file("quicklook.jpg")
        if quicklook is not None:
            return load_thumbnail(self._source, quicklook)
        return super().thumbnail()

    # -- geometry -------------------------------------------------------------
    def _read_metadata(self) -> Metadata:
        a = self._angles
        fields = {
            field: self._field(a["rows"], a["cols"], a[field], name=field, units="degrees")
            for field in files.ANGLES.values()
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
        shape = (self.height, self.width)
        return SceneGeometry(self._coarse_transformer, coarsen(self.metadata, shape, GEOMETRY_GRID), self._attrs, shape)

    @property
    def timestamp(self) -> datetime.datetime:
        return self._timestamp

    @property
    def footprint(self) -> dict:
        return self._footprint
