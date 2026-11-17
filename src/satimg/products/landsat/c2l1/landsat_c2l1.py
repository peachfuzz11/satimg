"""Landsat Collection-2 Level-1 reader."""

from __future__ import annotations

import math

import numpy
import rasterio
import xarray

from satimg.metadata.metadata import Metadata
from satimg.product import Product
from satimg.products.landsat.c2l1 import landsat_c2l1_parsing
from satimg.products.landsat.c2l1.landsat_c2l1_parsing import BANDS
from satimg.readers.raster import keep_open, merge_bands
from satimg.readers.thumbnail import load_thumbnail
from satimg.registry import register
from satimg.scene_geometry import SceneGeometry, coarsen
from satimg.source.source import Source
from satimg.tiling import as_band_yx, label_bands

#: side of the square grid :attr:`LandsatC2L1.geometry` stores each angle field on
GEOMETRY_GRID = 32

#: true-colour render, see :meth:`LandsatC2L1._render_visual`
WHITE_POINT = 0.35  #: top-of-atmosphere reflectance that maps to 255
GAMMA = 2.2
HAZE_FRACTION = 0.8  #: share of the scene's dark-object reflectance subtracted from each band
DARK_PERCENTILE = 0.5  #: percentile of a band's reflectance taken as its dark object
DARK_OBJECT_PX = 1024  #: longest side of the decimated read the dark object is taken from
PAN_WEIGHTS = (0.3, 0.3, 0.35)  #: red, green, blue weights of the synthetic panchromatic band


@register(r"^LC(0[1-9])_L1(TP|GT)_\d+_\d+_\d+_\d+_T1$")
class LandsatC2L1(Product):
    """``raw`` is every band resampled to the 15 m panchromatic grid; ``visual`` is
    a 3-band pan-sharpened true-colour render of top-of-atmosphere reflectance
    (see :meth:`_render_visual`). ``metadata`` carries the per-pixel
    angle rasters (``sun_zenith``, ``sun_azimuth``, ``view_zenith``,
    ``view_azimuth``), read decimated to a coarse grid. ``MTL.json``/angle-raster
    parsing itself lives in :mod:`satimg.products.landsat.c2l1.landsat_c2l1_parsing`."""

    @classmethod
    def scene_key(cls, name: str) -> str | None:
        return landsat_c2l1_parsing.scene_key(name)

    def __init__(self, source: Source):
        super().__init__(source)
        mtl = landsat_c2l1_parsing.parse_mtl(self._source)
        self._image_attrs = mtl["image_attrs"]
        self._timestamp = mtl["timestamp"]
        self._footprint = mtl["footprint"]
        self._transformer = mtl["transformer"]
        self._shape = mtl["shape"]
        self._reflectance = mtl["reflectance"]
        self._sin_sun = math.sin(math.radians(float(self._image_attrs["SUN_ELEVATION"])))
        self._dark: numpy.ndarray | None = None

    def _to_reflectance(self, band: str, dn):
        """Top-of-atmosphere reflectance of ``dn`` (a numpy array or DataArray), floored at 0."""
        mult, add = self._reflectance[band]
        return ((mult * dn + add) / self._sin_sun).clip(min=0)

    def _dark_object(self) -> numpy.ndarray:
        """``(red, green, blue)`` reflectance of the scene's haze: each band's
        :data:`DARK_PERCENTILE` over a decimated read, computed once per product so that every
        window and overview renders the same scene-level correction. 0 for a scene with no data."""
        if self._dark is None:
            levels = []
            for band in ("red", "green", "blue"):
                with rasterio.open(self._source.raster_path(f"{band}.TIF")) as src:
                    step = max(1, max(src.height, src.width) // DARK_OBJECT_PX)
                    # plain decimation: averaging would mix nodata into the scene's edge pixels
                    dn = src.read(1, out_shape=(src.height // step, src.width // step))
                valid = dn > 0
                levels.append(
                    float(numpy.percentile(self._to_reflectance(band, dn.astype("float32"))[valid], DARK_PERCENTILE))
                    if valid.any() else 0.0
                )
            self._dark = numpy.array(levels, dtype="float32")
        return self._dark

    def _open_raw(self, tile: int | tuple[int, int], overview: int = 1) -> xarray.DataArray:
        merged = merge_bands(self._visual_paths(), match=landsat_c2l1_parsing.PAN, tile=tile, overview=overview)
        return keep_open(label_bands(as_band_yx(merged), BANDS), merged)

    def _visual_paths(self) -> list[str]:
        # visual is pan-sharpened from every band of raw
        return [self._source.raster_path(f"{b}.TIF") for b in BANDS]

    def _render_visual(
        self, raw: xarray.DataArray, tile: int | tuple[int, int]
    ) -> xarray.DataArray:
        """True colour from top-of-atmosphere reflectance, pan-sharpened, haze-corrected.

        1. DN -> reflectance with ``MTL.json``'s rescaling and the sun elevation.
        2. Ratio-Brovey pan-sharpening: each colour keeps its own magnitude and is scaled by the
           pan band over a synthetic pan (:data:`PAN_WEIGHTS`).
        3. :data:`HAZE_FRACTION` of the scene's dark-object reflectance (:meth:`_dark_object`) is
           subtracted per band, the result is scaled so :data:`WHITE_POINT` maps to 255 and a
           :data:`GAMMA` curve is applied. Pixels with no pan data are 0.
        """
        a = raw.drop_vars("band")  # positional indexing below; relabel at the end
        bands = ("red", "green", "blue", "pan")
        dn = a.isel(band=[*landsat_c2l1_parsing.RGB, landsat_c2l1_parsing.PAN]).astype("float32")
        mult = numpy.array([self._reflectance[b][0] for b in bands], dtype="float32")[:, None, None]
        add = numpy.array([self._reflectance[b][1] for b in bands], dtype="float32")[:, None, None]
        rho = ((dn * mult + add) / self._sin_sun).clip(min=0)
        rgb, pan = rho.isel(band=slice(0, 3)), rho.isel(band=3)

        weights = numpy.array(PAN_WEIGHTS, dtype="float32")
        synthetic = (rgb * (weights / weights.sum())[:, None, None]).sum(dim="band")
        sharp = rgb * (pan / (synthetic + 1e-6)).clip(0, 4)

        low = (HAZE_FRACTION * self._dark_object())[:, None, None]
        span = numpy.maximum(WHITE_POINT - low, 1e-3)
        tone = ((sharp - low).clip(min=0) / span).clip(max=1) ** (1 / GAMMA)
        tone = tone.where(a.isel(band=landsat_c2l1_parsing.PAN) > 0, 0).fillna(0)
        return label_bands(as_band_yx((255 * tone).clip(0, 255).astype("uint8")),
                           ("red", "green", "blue"))

    def _read_metadata(self) -> Metadata:
        pan_transform = self._transformer.transform
        fields = {}
        for name, fname in landsat_c2l1_parsing.ANGLE_FILES.items():
            path = self._source.raster_path(fname)
            rows, cols, values = landsat_c2l1_parsing.read_angle_grid(path, pan_transform)
            fields[name] = self._field(rows, cols, values, name=name, units="degrees")

        want = ("SUN_ELEVATION", "SUN_AZIMUTH", "EARTH_SUN_DISTANCE", "ROLL_ANGLE")
        attrs = {
            k.lower(): float(self._image_attrs[k])
            for k in want
            if k in self._image_attrs
        }
        return Metadata(fields, attrs)

    @property
    def height(self) -> int:
        """From ``MTL.json``'s ``PANCHROMATIC_LINES`` (the pan grid ``raw`` is
        merged onto) rather than the base ``int(self.raw.sizes["y"])`` -- no
        ``raw`` open needed."""
        return self._shape[0]

    @property
    def width(self) -> int:
        """See :attr:`height`; from ``PANCHROMATIC_SAMPLES``."""
        return self._shape[1]

    @property
    def geometry(self) -> SceneGeometry:
        """As :attr:`Product.geometry`, with the angle fields resampled to a
        :data:`GEOMETRY_GRID` square grid -- the decimated rasters
        :attr:`metadata` reads are ~256 x 256 per field, too much to store
        per scene, and the angles vary smoothly. Like :attr:`metadata`,
        needs the product extracted."""
        shape = (self.height, self.width)
        return SceneGeometry(
            self.transformer, coarsen(self.metadata, shape, GEOMETRY_GRID), self.metadata.attrs, shape
        )

    def thumbnail(self):
        entries = self._source.listdir()
        for suffix in ("_small.jpeg", "thumbnail.jpeg", ".jpeg"):
            match = next((o for o in entries if o.endswith(suffix)), None)
            if match:
                return load_thumbnail(self._source, match)
        raise FileNotFoundError(f"no thumbnail under {self._source.name}")
