"""Sentinel-3 OLCI Level-1B full resolution (``OL_1_EFR___``) reader."""

from __future__ import annotations

import datetime
from functools import cached_property

import numpy
import PIL.Image
import xarray
from rasterio.control import GroundControlPoint

from satimg import olci_utils, swath
from satimg.metadata import Metadata
from satimg.product import Product
from satimg.readers import fit_thumbnail, keep_open, load_thumbnail, merge_bands, open_h5, vrt
from satimg.registry import register
from satimg.scene_geometry import SceneGeometry, coarsen
from satimg.source import Source
from satimg.tiling import as_band_yx, label_bands
from satimg.transform import SwathTransformer

#: metres the transformer may be off the tie-point geolocation between its
#: node rows (pixels are ~300 m)
TRANSFORMER_TOLERANCE_M = 10.0
#: ... and :attr:`OlciEFRProduct.geometry`'s stored one
GEOMETRY_TOLERANCE_M = 100.0
#: side of the square grid the stored geometry keeps each angle field on
GEOMETRY_GRID = 32

#: the visual's true colour bands (R, G, B) and the reflectance its stretch tops out at
_RGB = ("Oa08", "Oa06", "Oa04")
_VISUAL_MAX = 0.3


class OlciGeometry(SceneGeometry):
    """An OLCI product's :class:`~satimg.scene_geometry.SceneGeometry`: loads
    its transformer as a :class:`~satimg.transform.SwathTransformer`."""

    transformer_cls = SwathTransformer


@register(r"^S3[A-D]_OL_1_EFR____\d{8}T\d{6}_\d{8}T\d{6}_")
class OlciEFRProduct(Product):
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
    :mod:`satimg.olci_utils`): smooth, within about a third of a pixel of
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
            start, stop = olci_utils.read_times(h5)
            geolocation = olci_utils.read_geolocation(h5)
        with open_h5(self._source, f"{self._dir}{olci_utils.band_file('Oa01')}") as h5:
            self._shape = olci_utils.read_shape(h5)
        with open_h5(self._source, f"{self._dir}tie_geometries.nc") as h5:
            self._angles = olci_utils.read_angles(h5)
        scales, offsets = olci_utils.read_scales(self._source, self._dir)
        self._timestamp = start + (stop - start) / 2
        self._footprint = olci_utils.read_footprint(self._source) or swath.outline(geolocation["lat"], geolocation["lon"])
        self._attrs = {
            **olci_utils.read_name(self._source.name),
            "radiance_scales": scales,
            "radiance_offsets": offsets,
        }

        def transformer(tolerance):
            n = olci_utils.nodes(geolocation, tolerance)
            return SwathTransformer(n["rows"], n["cols"], n["lat"], n["lon"])

        self._transformer = transformer(TRANSFORMER_TOLERANCE_M)
        self._coarse_transformer = transformer(GEOMETRY_TOLERANCE_M)

    # -- pixels ---------------------------------------------------------------
    @cached_property
    def _band_paths(self) -> list[str]:
        """Each band through a VRT carrying a few GCPs -- just so GDAL sees
        it as georeferenced; :attr:`transformer` is what maps it."""
        rows = numpy.linspace(0, self.height, 5)
        cols = numpy.linspace(0, self.width, 5)
        rr, cc = (a.ravel() for a in numpy.meshgrid(rows, cols, indexing="ij"))
        latlon = self._coarse_transformer.rowcol_to_latlon(numpy.column_stack((rr, cc)))
        gcps = [GroundControlPoint(row=r, col=c, x=lon, y=lat) for r, c, (lat, lon) in zip(rr, cc, latlon)]
        return [
            vrt(olci_utils.gdal_path(self._source.raster_path(f"{self._dir}{olci_utils.band_file(b)}"), b),
                self._shape, gcps=gcps, dtype="UInt16", nodata=65535)
            for b in olci_utils.BANDS
        ]

    def _visual_paths(self) -> list[str]:
        return []  # no overviews in netCDF

    def _open_raw(self, tile: int | tuple[int, int], overview: int = 1) -> xarray.DataArray:
        if overview != 1:
            raise ValueError("an OLCI product has no overviews")
        merged = merge_bands(self._band_paths, tile=tile)
        da = label_bands(as_band_yx(merged), olci_utils.BANDS)
        da = da.assign_coords(wavelength_nm=("band", list(olci_utils.WAVELENGTHS_NM)))
        return keep_open(da, merged)

    def radiance(self, raw: xarray.DataArray) -> xarray.DataArray:
        """``raw`` (or a band / window of it) in mW / (m^2 sr nm),
        ``float32``, fills ``NaN``."""
        pick = [olci_utils.BANDS.index(b) for b in raw.band.values]
        scales = numpy.array(self._attrs["radiance_scales"], dtype="float32")[pick, None, None]
        offsets = numpy.array(self._attrs["radiance_offsets"], dtype="float32")[pick, None, None]
        return raw.where(raw != 65535).astype("float32") * scales + offsets

    def _render_visual(
        self, raw: xarray.DataArray, tile: int | tuple[int, int]
    ) -> xarray.DataArray:
        rgb = self.radiance(raw.sel(band=list(_RGB)))
        e0 = numpy.array([olci_utils.SOLAR_IRRADIANCE[b] for b in _RGB], dtype="float32")[:, None, None]
        reflectance = numpy.pi * rgb / e0  # x cos(sun zenith), left out: fine for a picture
        u8 = (255 * (reflectance.clip(0, _VISUAL_MAX) / _VISUAL_MAX) ** (1 / 2.2)).fillna(0).astype("uint8")
        return label_bands(as_band_yx(u8.drop_vars("wavelength_nm", errors="ignore")), ("red", "green", "blue"))

    def thumbnail(self) -> PIL.Image.Image:
        """The product's shipped ``quicklook.jpg``, else rendered from
        :attr:`visual`, decimated."""
        quicklook = self._source.find_file("quicklook.jpg")
        if quicklook is not None:
            return load_thumbnail(self._source, quicklook)
        step = max(1, max(self._shape) // 400)
        return fit_thumbnail(PIL.Image.fromarray(numpy.moveaxis(self.visual[:, ::step, ::step].values, 0, -1)))

    # -- geometry -------------------------------------------------------------
    def _read_metadata(self) -> Metadata:
        a = self._angles
        fields = {
            field: self._field(a["rows"], a["cols"], a[field], name=field, units="degrees")
            for field in olci_utils.ANGLES.values()
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
    def geometry(self) -> OlciGeometry:
        shape = (self.height, self.width)
        return OlciGeometry(self._coarse_transformer, coarsen(self.metadata, shape, GEOMETRY_GRID), self._attrs, shape)

    @property
    def timestamp(self) -> datetime.datetime:
        return self._timestamp

    @property
    def footprint(self) -> dict:
        return self._footprint
