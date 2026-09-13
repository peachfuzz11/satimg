"""Pixel <-> geographic coordinate conversion.

:class:`Transformer` wraps an affine ``transform`` + ``crs`` (the common case).
:class:`GCPTransformer` does the same job for products that only ship ground
control points (Sentinel-1), via bilinear interpolation on the (rectilinear)
GCP grid.

Both accept a single ``(a, b)`` pair, a list of pairs, or an ``(N, 2)`` array and
always return an ``(N, 2)`` array. ``rowcol`` is ``(row, col)`` = ``(y, x)``;
``latlon`` is ``(lat, lon)``.

Built entirely from product metadata (affine origin/spacing, EPSG code, or a
geolocation-grid-point list) -- no rasterio/rioxarray dataset needs to be opened.
"""

from __future__ import annotations

from functools import cached_property
from typing import Sequence, Union

import numpy
import pyproj
from affine import Affine

Coords = Union[Sequence[float], Sequence[Sequence[float]], numpy.ndarray]


def _as_n2(coords: Coords) -> numpy.ndarray:
    arr = numpy.array(coords, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, 2)
    return arr


class Transformer:
    """Affine pixel/CRS transform plus reprojection to and from EPSG:4326."""

    def __init__(self, transform: Affine, crs: pyproj.CRS):
        self._transform = transform
        self._crs = pyproj.CRS.from_user_input(crs)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(crs={self._crs}, transform={self._transform})"

    @cached_property
    def _to_wgs84(self) -> pyproj.Transformer:
        return pyproj.Transformer.from_crs(self._crs, "EPSG:4326", always_xy=True)

    def rowcol_to_latlon(self, coords: Coords) -> numpy.ndarray:
        rc = _as_n2(coords)
        xs, ys = self._transform * (rc[:, 1], rc[:, 0])
        lons, lats = numpy.asarray(xs), numpy.asarray(ys)
        if not self._crs.is_geographic:
            lons, lats = self._to_wgs84.transform(lons.tolist(), lats.tolist())
            lons, lats = numpy.asarray(lons), numpy.asarray(lats)
        return numpy.column_stack((lats, lons))

    def latlon_to_rowcol(self, coords: Coords) -> numpy.ndarray:
        ll = _as_n2(coords)
        lats, lons = ll[:, 0], ll[:, 1]
        if not self._crs.is_geographic:
            xs, ys = self._to_wgs84.transform(lons.tolist(), lats.tolist(), direction="INVERSE")
            lons, lats = numpy.asarray(xs), numpy.asarray(ys)
        cols, rows = ~self._transform * (lons, lats)
        return numpy.column_stack((numpy.asarray(rows), numpy.asarray(cols)))


class GCPTransformer(Transformer):
    """Coordinate transform backed by a rectilinear grid of ground control
    points (e.g. Sentinel-1's ``geolocationGridPoint`` list), in geographic
    (lat/lon) coordinates. Forward lookup is bilinear interpolation on the
    grid; the inverse is a nearest-point guess refined by a few Newton
    iterations, since bilinear interpolation has no closed-form inverse.
    """

    #: Newton iteration count and finite-difference step (grid pixels) for the inverse.
    _NEWTON_ITERS = 6
    _FD_STEP = 1e-3

    def __init__(self, rows, cols, lat_grid, lon_grid):
        self._rows = numpy.asarray(rows, dtype=float)
        self._cols = numpy.asarray(cols, dtype=float)
        self._lat_grid = numpy.asarray(lat_grid, dtype=float)
        self._lon_grid = numpy.asarray(lon_grid, dtype=float)
        self._crs = pyproj.CRS.from_epsg(4326)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(grid={len(self._rows)}x{len(self._cols)})"

    def _bilinear(self, rc: numpy.ndarray) -> numpy.ndarray:
        from satimg.metadata import bilinear

        lat = bilinear(self._rows, self._cols, self._lat_grid, rc[:, 0], rc[:, 1])
        lon = bilinear(self._rows, self._cols, self._lon_grid, rc[:, 0], rc[:, 1])
        return numpy.column_stack((lat, lon))

    def rowcol_to_latlon(self, coords: Coords) -> numpy.ndarray:
        return self._bilinear(_as_n2(coords))

    def _nearest_rowcol(self, ll: numpy.ndarray) -> numpy.ndarray:
        """Nearest GCP grid point to each ``(lat, lon)``, as ``(row, col)``."""
        rr, cc = numpy.meshgrid(self._rows, self._cols, indexing="ij")
        lat, lon = self._lat_grid.ravel(), self._lon_grid.ravel()
        rr, cc = rr.ravel(), cc.ravel()
        out = numpy.empty((len(ll), 2))
        for i, (target_lat, target_lon) in enumerate(ll):
            d2 = (lat - target_lat) ** 2 + (lon - target_lon) ** 2
            j = numpy.argmin(d2)
            out[i] = (rr[j], cc[j])
        return out

    def latlon_to_rowcol(self, coords: Coords) -> numpy.ndarray:
        ll = _as_n2(coords)
        rc = self._nearest_rowcol(ll)
        step = self._FD_STEP
        for _ in range(self._NEWTON_ITERS):
            f0 = self._bilinear(rc) - ll
            f_dr = self._bilinear(rc + [step, 0]) - ll
            f_dc = self._bilinear(rc + [0, step]) - ll
            j00 = (f_dr[:, 0] - f0[:, 0]) / step
            j01 = (f_dc[:, 0] - f0[:, 0]) / step
            j10 = (f_dr[:, 1] - f0[:, 1]) / step
            j11 = (f_dc[:, 1] - f0[:, 1]) / step
            det = j00 * j11 - j01 * j10
            det = numpy.where(det == 0, numpy.finfo(float).eps, det)
            d_row = (j11 * f0[:, 0] - j01 * f0[:, 1]) / det
            d_col = (j00 * f0[:, 1] - j10 * f0[:, 0]) / det
            rc = rc - numpy.column_stack((d_row, d_col))
        return rc
