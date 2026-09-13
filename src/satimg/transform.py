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


def _invert_bilinear_cell(p, a, b, c, d):
    """Inverse of the bilinear map from the unit square to the quad
    ``a, b, c, d`` (corners at ``(row, col)`` fractions ``(0, 0)``,
    ``(0, 1)``, ``(1, 1)``, ``(1, 0)``): the ``(tr, tc)`` such that
    bilinearly interpolating those corners at ``(tr, tc)`` gives ``p``, or
    ``None`` if ``p`` is outside the quad.

    Solved by Newton's method with the exact (analytic, not finite-difference)
    Jacobian, from the cell's centre. A GCP cell is only mildly non-affine
    (its bilinear "twist" term is a couple of percent of its edges, for
    Sentinel-1's geolocation grid), so this converges to machine precision in
    2-3 iterations every time -- no discriminants or degenerate cases to
    special-case.
    """
    e, f, g = b - a, d - a, a - b + c - d  # p(tr, tc) = a + tc*e + tr*f + tr*tc*g
    tr, tc = 0.5, 0.5
    for _ in range(5):
        residual = p - (a + tc * e + tr * f + tr * tc * g)
        jacobian = numpy.array([[f[0] + tc * g[0], e[0] + tr * g[0]],
                                 [f[1] + tc * g[1], e[1] + tr * g[1]]])
        try:
            d_tr, d_tc = numpy.linalg.solve(jacobian, residual)
        except numpy.linalg.LinAlgError:
            return None
        tr, tc = tr + d_tr, tc + d_tc

    if -1e-6 <= tr <= 1 + 1e-6 and -1e-6 <= tc <= 1 + 1e-6:
        return numpy.clip(tr, 0, 1), numpy.clip(tc, 0, 1)
    return None


class GCPTransformer(Transformer):
    """Coordinate transform backed by a rectilinear grid of ground control
    points (e.g. Sentinel-1's ``geolocationGridPoint`` list), in geographic
    (lat/lon) coordinates. Forward lookup is bilinear interpolation on the
    grid; the inverse locates the grid cell containing the target point and
    inverts the bilinear map for that cell (:func:`_invert_bilinear_cell`).
    """

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
        rows, cols = self._rows, self._cols
        lat, lon = self._lat_grid, self._lon_grid

        # per-cell bounding boxes, to shortlist which cell each point falls in
        corners_lat = (lat[:-1, :-1], lat[:-1, 1:], lat[1:, :-1], lat[1:, 1:])
        corners_lon = (lon[:-1, :-1], lon[:-1, 1:], lon[1:, :-1], lon[1:, 1:])
        lat_lo, lat_hi = numpy.minimum.reduce(corners_lat), numpy.maximum.reduce(corners_lat)
        lon_lo, lon_hi = numpy.minimum.reduce(corners_lon), numpy.maximum.reduce(corners_lon)

        out = numpy.empty_like(ll)
        for k, (target_lat, target_lon) in enumerate(ll):
            p = numpy.array([target_lat, target_lon])
            found = None
            for i, j in zip(*numpy.where(
                (lat_lo <= target_lat) & (target_lat <= lat_hi)
                & (lon_lo <= target_lon) & (target_lon <= lon_hi)
            )):
                a = numpy.array([lat[i, j], lon[i, j]])
                b = numpy.array([lat[i, j + 1], lon[i, j + 1]])
                c = numpy.array([lat[i + 1, j + 1], lon[i + 1, j + 1]])
                d = numpy.array([lat[i + 1, j], lon[i + 1, j]])
                tr_tc = _invert_bilinear_cell(p, a, b, c, d)
                if tr_tc is not None:
                    tr, tc = tr_tc
                    found = (
                        rows[i] + tr * (rows[i + 1] - rows[i]),
                        cols[j] + tc * (cols[j + 1] - cols[j]),
                    )
                    break
            out[k] = found if found is not None else self._nearest_rowcol(ll[k : k + 1])[0]
        return out
