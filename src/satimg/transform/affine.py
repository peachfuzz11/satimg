"""Affine georeferencing: a map-projected raster's geotransform and CRS."""

from __future__ import annotations

import numpy
from rasterio import warp
from rasterio.crs import CRS
from rasterio.transform import Affine

from satimg.transform.base import Coords, Transformer, as_n2


class AffineTransformer(Transformer):
    """An affine pixel -> CRS ``transform`` plus reprojection to and from
    EPSG:4326 -- for map-projected products (Sentinel-2, Landsat, NISAR
    GCOV)."""

    key = "transform"

    def __init__(self, transform: Affine, crs: CRS):
        self._transform = transform
        self._crs = crs

    def __repr__(self) -> str:
        return f"{type(self).__name__}(crs={self._crs}, transform={self._transform})"

    @property
    def transform(self) -> Affine:
        return self._transform

    @property
    def crs(self) -> CRS:
        return self._crs

    def to_dict(self) -> dict:
        return {"transform": list(self._transform)[:6], "crs": self._crs.to_wkt()}

    @classmethod
    def _from_dict(cls, data: dict) -> AffineTransformer:
        return cls(Affine(*data["transform"]), CRS.from_wkt(data["crs"]))

    def rowcol_to_latlon(self, coords: Coords) -> numpy.ndarray:
        rc = as_n2(coords)
        xs, ys = self._transform * (rc[:, 1], rc[:, 0])
        lons, lats = numpy.asarray(xs), numpy.asarray(ys)
        if not self._crs.is_geographic:
            lons, lats = warp.transform(self._crs, "EPSG:4326", lons.tolist(), lats.tolist())
            lons, lats = numpy.asarray(lons), numpy.asarray(lats)
        return numpy.column_stack((lats, lons))

    def latlon_to_rowcol(self, coords: Coords) -> numpy.ndarray:
        ll = as_n2(coords)
        lats, lons = ll[:, 0], ll[:, 1]
        if not self._crs.is_geographic:
            xs, ys = warp.transform("EPSG:4326", self._crs, lons.tolist(), lats.tolist())
            lons, lats = numpy.asarray(xs), numpy.asarray(ys)
        cols, rows = ~self._transform * (lons, lats)
        return numpy.column_stack((numpy.asarray(rows), numpy.asarray(cols)))
