"""MODIS Collection 7 L1B 250 m reader (``MOD02QKM`` Terra, ``MYD02QKM`` Aqua)."""

from __future__ import annotations

import datetime

import numpy
import PIL.Image
import xarray

from satimg.metadata.collection import Metadata
from satimg.product import Product
from satimg.products import swath
from satimg.products.modis import files
from satimg.readers.hdf5 import find_dataset, open_h5, open_h5_array
from satimg.readers.thumbnail import fit_thumbnail
from satimg.registry import register
from satimg.scene_geometry import SceneGeometry
from satimg.source.base import Source
from satimg.tiling import as_band_yx, label_bands
from satimg.transform.swath import SwathTransformer

#: metres the transformer may be off the 1 km geolocation between its nodes
TRANSFORMER_TOLERANCE_M = 30.0
#: ... and :attr:`ModisProduct.geometry`'s stored one
GEOMETRY_TOLERANCE_M = 125.0

#: TOA reflectance (x cos sun zenith) the visual's stretch tops out at --
#: water is a few percent in the NIR, a ship or cloud far brighter
_VISUAL_MAX = 0.4



@register(r"^M[OY]D02QKM(_NRT)?\.A\d{7}\.\d{4}\.\d{3}")
class ModisProduct(Product):
    """One MODIS 250 m L1B granule -- Terra (``MOD02QKM``) or Aqua
    (``MYD02QKM``), Collection 7 netCDF4: five minutes of orbit, 8120 x 5416
    pixels in sensor (swath) geometry. Open the ``.nc`` itself, or a folder /
    zip holding it with its ``MOD03`` / ``MYD03`` geolocation, as the
    ``modis-qkm`` connector downloads. The geolocation file supplies the
    1 km positions and ``metadata``'s sun / view angles (``solar_zenith``,
    ``solar_azimuth``, ``satellite_zenith``, ``satellite_azimuth``); without
    it the positions the 250 m file carries are used and there are no
    angles.

    ``raw`` is the scaled integers as stored (``uint16``, bands ``B01`` red
    and ``B02`` NIR, ``wavelength_nm`` on the band axis; anything above 32767
    is a fill or flag) -- ``metadata.attrs`` has each band's
    ``reflectance_scales`` / ``reflectance_offsets`` and ``radiance_*`` to
    calibrate them (reflectance x cos(sun zenith) = scale x (DN - offset)).
    ``visual`` is the NIR band as greyscale, where open water is darkest and
    ships, wakes and clouds stand out. Only day-mode granules carry 250 m
    data; a night granule's ``raw`` is all fill.

    The transformer interpolates the 1 km geolocation scan by scan (MODIS's
    scans overlap at the swath edge, as VIIRS's do), its nodes placed with
    the guide's fractional offset (1.5 pixels down track). At 250 m only
    large ships and wakes are visible, and the image models were not trained
    on MODIS.
    """

    def __init__(self, source: Source):
        super().__init__(source)
        self._file = files.find_file(self._source)
        self._geo_file = files.find_geolocation(self._source, self._file)
        with open_h5(self._source, self._file) as h5:
            granule = files.read_granule(h5, self._file)
            offsets = files.fractional_offsets(h5)
            self._dataset = find_dataset(h5, files.DATA)
            if self._geo_file is None:
                geolocation = files.read_geolocation(h5)
        self._timestamp = granule["timestamp"]
        self._attrs = granule["attrs"]
        self._shape = granule["shape"]

        def nodes(geolocation, tolerance):
            return files.nodes(geolocation, offsets, tolerance)

        if self._geo_file is None:
            coarse = nodes(geolocation, GEOMETRY_TOLERANCE_M)
        else:
            with open_h5(self._source, self._geo_file) as h5:
                geolocation = files.read_geolocation(h5)
                coarse = nodes(geolocation, GEOMETRY_TOLERANCE_M)
                coarse.update(files.read_angles(h5, coarse))
        # the scan-edge rows include the first and last, so the outline still
        # runs round the whole swath
        self._footprint = swath.outline(geolocation["lat"], geolocation["lon"])
        fine = nodes(geolocation, TRANSFORMER_TOLERANCE_M)
        self._transformer = SwathTransformer(
            fine["rows"], fine["cols"], fine["lat"], fine["lon"], scan_rows=files.SCAN_ROWS
        )
        self._coarse_transformer = SwathTransformer(
            coarse["rows"], coarse["cols"], coarse["lat"], coarse["lon"], scan_rows=files.SCAN_ROWS
        )
        self._angles = coarse

    # -- pixels ---------------------------------------------------------------
    def _open_raw(self, tile: int | tuple[int, int], overview: int = 1) -> xarray.DataArray:
        if overview != 1:
            raise ValueError("a MODIS granule has no overviews")
        da = open_h5_array(self._source, self._file, self._dataset, tile)
        out = label_bands(as_band_yx(da), files.BANDS)
        out = out.assign_coords(wavelength_nm=("band", list(files.WAVELENGTHS_NM)))
        out.set_close(da._close)
        return out

    def reflectance(self, raw: xarray.DataArray) -> xarray.DataArray:
        """``raw`` (or a slice of it) as TOA reflectance x cos(sun zenith),
        ``float32``, fills and flags ``NaN``."""
        scales = numpy.array(self._attrs["reflectance_scales"], dtype="float32")
        offsets = numpy.array(self._attrs["reflectance_offsets"], dtype="float32")
        pick = [files.BANDS.index(b) for b in raw.band.values]
        dn = raw.where(raw <= files.VALID_MAX).astype("float32")
        return (dn - offsets[pick, None, None]) * scales[pick, None, None]

    def _render_visual(
        self, raw: xarray.DataArray, tile: int | tuple[int, int]
    ) -> xarray.DataArray:
        nir = self.reflectance(raw.sel(band=["B02"])).isel(band=0)
        u8 = (255 * (nir.clip(0, _VISUAL_MAX) / _VISUAL_MAX) ** (1 / 2.2)).fillna(0).astype("uint8")
        return label_bands(as_band_yx(u8), ("nir",))

    def thumbnail(self) -> PIL.Image.Image:
        """Rendered from :attr:`visual`, decimated -- the file ships no
        quick-look."""
        step = max(1, max(self._shape) // 400)
        return fit_thumbnail(PIL.Image.fromarray(self.visual[0, ::step, ::step].values, mode="L"))

    # -- geometry -------------------------------------------------------------
    def _read_metadata(self) -> Metadata:
        """The sun / view angles from the ``M?D03`` file, when it is
        alongside (the 250 m file has none); calibration and scene facts in
        ``attrs``."""
        a = self._angles
        fields = {
            field: self._field(a["rows"], a["cols"], a[field], name=field, units=units)
            for field, units in files.ANGLES.values()
            if field in a
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
        m = self.metadata
        fields = {name: (m[name]._rows, m[name]._cols, m[name]._values, m[name].units) for name in m}
        return SceneGeometry(self._coarse_transformer, fields, self._attrs, self._shape)

    @property
    def timestamp(self) -> datetime.datetime:
        return self._timestamp

    @property
    def footprint(self) -> dict:
        return self._footprint
