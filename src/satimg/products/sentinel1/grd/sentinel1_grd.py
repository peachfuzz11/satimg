"""Sentinel-1 GRD (IW and EW) reader."""

from __future__ import annotations

import os
import re
from functools import cached_property

import numpy
import PIL.Image
import xarray

from satimg.metadata.metadata import Metadata
from satimg.products.sentinel1.grd import sentinel1_grd_parsing
from satimg.readers.raster import keep_open, merge_bands
from satimg.readers.thumbnail import load_thumbnail
from satimg.registry import register
from satimg.sar_orbit_geometry import SAROrbitGeometry
from satimg.sar_product import SARProduct
from satimg.source.source import Source
from satimg.tiling import as_band_yx, label_bands


@register(r"^S1[ABCD]_(IW_GRDH|EW_GRDM)_1SD[HV]_\d{8}T\d{6}_\d{8}T\d{6}.*\.SAFE$")
class Sentinel1GRD(SARProduct):
    """Ground-range-detected Sentinel-1, either acquisition mode.

    ``raw`` holds the calibrated backscatter (one band per polarisation);
    ``visual`` is the usual dB -> sigmoid stretch as a single greyscale band.
    ``metadata`` carries the ``geolocationGridPoint`` fields (``incidence_angle``,
    ``elevation_angle``, ``slant_range_time``, ``height``) and its ``attrs`` the
    annotation's orbit state vectors (``orbit``), which the moving-target
    correction and heading conversions (:class:`~satimg.sar_product.SARProduct`)
    work from. SAFE parsing itself lives in
    :mod:`satimg.products.sentinel1.grd.sentinel1_grd_parsing`.
    """

    def __init__(self, source: Source):
        super().__init__(source)
        meta = sentinel1_grd_parsing.read_manifest(self._source)
        self._timestamp = meta["timestamp"]
        self._footprint = meta["footprint"]
        annotation = sentinel1_grd_parsing.annotation_file(self._source)
        points, attrs, shape = sentinel1_grd_parsing.read_geolocation(
            self._source, annotation, self._timestamp
        )
        self._geometry = SAROrbitGeometry(
            sentinel1_grd_parsing.build_transformer(points), sentinel1_grd_parsing.build_fields(points),
            attrs, shape,
        )

    @property
    def mode(self) -> str:
        m = re.search(r"_(IW|EW)_", os.path.basename(self._path))
        return m.group(1) if m else "IW"

    @cached_property
    def _measurements(self) -> dict[str, str]:
        """Each measurement raster's path to its polarisation, in band order."""
        pols = {}
        for f in self._source.listdir("measurement"):
            m = re.search(r"-(vv|vh|hh|hv)-", f)
            if m:
                pols[self._source.raster_path(f"measurement/{f}")] = m.group(1)
        return {f: pols[f] for f in sorted(pols, key=lambda f: sentinel1_grd_parsing.POL_ORDER[pols[f]])}

    def _open_raw(self, tile: int | tuple[int, int], overview: int = 1) -> xarray.DataArray:
        merged = merge_bands(list(self._measurements), tile=tile, overview=overview)
        da = label_bands(as_band_yx(merged), [pol.upper() for pol in self._measurements.values()])
        return keep_open(da, merged)

    def _visual_paths(self) -> list[str]:
        return list(self._measurements)

    def _render_visual(
        self, raw: xarray.DataArray, tile: int | tuple[int, int]
    ) -> xarray.DataArray:
        db = (10 * numpy.log10(raw.where(raw > 0))).fillna(0).mean("band")
        u8 = (255 / (1 + numpy.exp(-((db - 20) * 0.18)))).clip(0, 255).astype("uint8")
        return label_bands(as_band_yx(u8), ("amplitude",))

    def _read_metadata(self) -> Metadata:
        return self._geometry.metadata

    def thumbnail(self) -> PIL.Image.Image:
        """The shipped preview PNG, forced to greyscale: SAR quick-looks ship
        as an RGB polarimetric composite (e.g. VV/VH/ratio as R/G/B) on some
        products, but the plain, dark, single-band amplitude look -- matching
        :attr:`visual` -- is the more useful thumbnail for a SAR scene."""
        for name in ("thumbnail.png", "quick-look.png"):
            relpath = f"preview/{name}"
            if self._source.exists(relpath):
                return load_thumbnail(self._source, relpath, greyscale=True)
        raise FileNotFoundError(f"no preview image under {self._source.name}")
