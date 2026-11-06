"""Pixel <-> geographic coordinate conversion.

:class:`Transformer` wraps an affine ``transform`` + ``crs`` (the common case).
:class:`GCPTransformer` does the same job for products that only ship ground
control points (Sentinel-1).

Both accept a single ``(a, b)`` pair, a list of pairs, or an ``(N, 2)`` array and
always return an ``(N, 2)`` array. ``rowcol`` is ``(row, col)`` = ``(y, x)``;
``latlon`` is ``(lat, lon)``.

Each round-trips through a JSON-serialisable dict of its own (``to_dict`` /
``from_dict``), so a product's georeferencing can be stored and used after the
product file is gone.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy
import rasterio
from rasterio import warp
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
from rasterio.transform import Affine

Coords = Sequence[float] | Sequence[Sequence[float]] | numpy.ndarray


def _as_n2(coords: Coords) -> numpy.ndarray:
    arr = numpy.array(coords, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, 2)
    return arr


class Transformer:
    """Affine pixel/CRS transform plus reprojection to and from EPSG:4326."""

    def __init__(self, transform, crs):
        self._transform = transform
        self._crs = crs

    def __repr__(self) -> str:
        return f"{type(self).__name__}(crs={self._crs}, transform={self._transform})"

    def to_dict(self) -> dict:
        """A JSON-serialisable dict that :meth:`from_dict` turns back into an
        equal transformer."""
        return {"transform": list(self._transform)[:6], "crs": self._crs.to_wkt()}

    @classmethod
    def from_dict(cls, data: dict) -> Transformer:
        return cls(Affine(*data["transform"]), CRS.from_wkt(data["crs"]))

    def rowcol_to_latlon(self, coords: Coords) -> numpy.ndarray:
        rc = _as_n2(coords)
        xs, ys = self._transform * (rc[:, 1], rc[:, 0])
        lons, lats = numpy.asarray(xs), numpy.asarray(ys)
        if not self._crs.is_geographic:
            lons, lats = warp.transform(self._crs, "EPSG:4326", lons.tolist(), lats.tolist())
            lons, lats = numpy.asarray(lons), numpy.asarray(lats)
        return numpy.column_stack((lats, lons))

    def latlon_to_rowcol(self, coords: Coords) -> numpy.ndarray:
        ll = _as_n2(coords)
        lats, lons = ll[:, 0], ll[:, 1]
        if not self._crs.is_geographic:
            xs, ys = warp.transform("EPSG:4326", self._crs, lons.tolist(), lats.tolist())
            lons, lats = numpy.asarray(xs), numpy.asarray(ys)
        cols, rows = ~self._transform * (lons, lats)
        return numpy.column_stack((numpy.asarray(rows), numpy.asarray(cols)))

    def north_angle(self, coords: Coords) -> numpy.ndarray:
        """The direction true north points in the image at each ``(row, col)``,
        in degrees clockwise from image "up" (decreasing row), ``[0, 360)``
        -- an ``(N,)`` array.

        ``0`` for a north-up geographic grid; a few degrees either way across
        a UTM zone (grid convergence); anything at all on a polar
        stereographic grid. Found by stepping a short way north from each
        point and seeing where that lands in pixels, so it holds for any CRS.
        (Not for a :class:`GCPTransformer`, whose ``latlon_to_rowcol`` snaps
        to whole pixels.)
        """
        rc = _as_n2(coords)
        latlon = self.rowcol_to_latlon(rc)
        # ~10 m north; towards the equator instead right at a pole
        step = numpy.where(latlon[:, 0] > 89.9, -1e-4, 1e-4)
        ahead = self.latlon_to_rowcol(numpy.column_stack((latlon[:, 0] + step, latlon[:, 1])))
        d_row, d_col = (ahead - rc).T * numpy.sign(step)
        return numpy.rad2deg(numpy.arctan2(d_col, -d_row)) % 360.0


class GCPTransformer(Transformer):
    """Coordinate transform backed by ground control points."""

    def __init__(self, gcps, crs):
        super().__init__(rasterio.transform.GCPTransformer(gcps), crs)
        self._gcps = list(gcps)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(crs={self._crs}, gcps={len(self._gcps)})"

    def to_dict(self) -> dict:
        return {
            "gcps": [[g.row, g.col, g.x, g.y, g.z] for g in self._gcps],
            "crs": self._crs.to_wkt(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> GCPTransformer:
        gcps = [GroundControlPoint(row=r, col=c, x=x, y=y, z=z) for r, c, x, y, z in data["gcps"]]
        return cls(gcps, CRS.from_wkt(data["crs"]))

    def rowcol_to_latlon(self, coords: Coords) -> numpy.ndarray:
        rc = _as_n2(coords)
        lons, lats = self._transform.xy(rc[:, 0].tolist(), rc[:, 1].tolist(), offset="center")
        return numpy.column_stack((numpy.asarray(lats), numpy.asarray(lons)))

    def latlon_to_rowcol(self, coords: Coords) -> numpy.ndarray:
        ll = _as_n2(coords)
        rows, cols = self._transform.rowcol(ll[:, 1].tolist(), ll[:, 0].tolist())
        return numpy.column_stack((numpy.asarray(rows), numpy.asarray(cols)))


def _unit_vectors(lat, lon) -> numpy.ndarray:
    """``(..., 3)`` earth-centred unit vectors of ``lat`` / ``lon`` degrees."""
    lat, lon = numpy.deg2rad(lat), numpy.deg2rad(lon)
    return numpy.stack([numpy.cos(lat) * numpy.cos(lon), numpy.cos(lat) * numpy.sin(lon), numpy.sin(lat)], axis=-1)


class SwathTransformer(Transformer):
    """Coordinate transform for a sensor-geometry swath (VIIRS) that ships a
    latitude / longitude for every pixel: those arrays kept on a coarse
    ``rows`` x ``cols`` node grid (pixel coordinates, pixel centres at
    ``i + 0.5``) and interpolated -- bilinearly, on the unit sphere, so the
    antimeridian and the poles need no special casing.

    A whisk-broom scanner images ``scan_rows`` rows at a time, and its
    geolocation jumps from one scan to the next (VIIRS's "bow-tie": scans
    overlap towards the swath edges). With ``scan_rows`` set, a pixel is only
    ever interpolated between its own scan's nodes -- extrapolated to the
    scan's first / last pixel edge, never across into the next scan.

    ``latlon_to_rowcol`` inverts that by Gauss-Newton inside the scans around
    the nearest node, keeping the best fit; a point off the swath -- further
    than a pixel from any pixel -- comes back ``NaN``. Where scans overlap
    either of the pixels seeing a point may be returned.
    """

    def __init__(self, rows, cols, lat, lon, scan_rows: int | None = None):
        self._rows = numpy.asarray(rows, dtype=float)
        self._cols = numpy.asarray(cols, dtype=float)
        self._lat = numpy.asarray(lat, dtype=float)
        self._lon = numpy.asarray(lon, dtype=float)
        self._xyz = _unit_vectors(self._lat, self._lon)
        self._scan_rows = scan_rows
        self._transform = None
        self._crs = CRS.from_epsg(4326)
        # node index range [first, last] of each scan (one scan: everything)
        if scan_rows:
            scan_of_node = (self._rows // scan_rows).astype(int)
            n_scans = int(scan_of_node[-1]) + 1
            self._first = numpy.searchsorted(scan_of_node, numpy.arange(n_scans), side="left")
            self._last = numpy.searchsorted(scan_of_node, numpy.arange(n_scans), side="right") - 1
        else:
            self._first, self._last = numpy.array([0]), numpy.array([len(self._rows) - 1])

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nodes={len(self._rows)}x{len(self._cols)}, scan_rows={self._scan_rows})"

    def to_dict(self) -> dict:
        # 5 decimals of a degree is ~1 m: plenty, at a third of the JSON
        return {
            "swath": {
                "rows": self._rows.tolist(),
                "cols": self._cols.tolist(),
                "lat": numpy.round(self._lat, 5).tolist(),
                "lon": numpy.round(self._lon, 5).tolist(),
                "scan_rows": self._scan_rows,
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> SwathTransformer:
        swath = data["swath"]
        return cls(swath["rows"], swath["cols"], swath["lat"], swath["lon"], swath.get("scan_rows"))

    def _scan(self, rows) -> numpy.ndarray:
        if not self._scan_rows:
            return numpy.zeros(numpy.shape(rows), dtype=int)
        return numpy.clip(numpy.floor(rows / self._scan_rows).astype(int), 0, len(self._first) - 1)

    def _xyz_at(self, rows, cols, scans=None) -> numpy.ndarray:
        """Unit vectors at ``rows`` / ``cols``, interpolated within each
        point's scan (``scans``, else the one its row falls in); linearly
        extrapolated past the outer nodes rather than clamped, so pixels
        between the outermost node centre and an edge still move."""
        rows, cols = numpy.asarray(rows, dtype=float), numpy.asarray(cols, dtype=float)
        scans = self._scan(rows) if scans is None else scans
        first, last = self._first[scans], self._last[scans]
        i = numpy.clip(numpy.searchsorted(self._rows, rows, side="right") - 1, first, numpy.maximum(last - 1, first))
        i2 = numpy.minimum(i + 1, last)
        j = numpy.clip(numpy.searchsorted(self._cols, cols, side="right") - 1, 0, len(self._cols) - 2)
        span = self._rows[i2] - self._rows[i]
        tr = numpy.where(span > 0, (rows - self._rows[i]) / numpy.where(span > 0, span, 1.0), 0.0)[:, None]
        tc = ((cols - self._cols[j]) / (self._cols[j + 1] - self._cols[j]))[:, None]
        g = self._xyz
        top = g[i, j] * (1 - tc) + g[i, j + 1] * tc
        bottom = g[i2, j] * (1 - tc) + g[i2, j + 1] * tc
        v = top * (1 - tr) + bottom * tr
        return v / numpy.linalg.norm(v, axis=-1, keepdims=True)

    def rowcol_to_latlon(self, coords: Coords) -> numpy.ndarray:
        rc = _as_n2(coords)
        v = self._xyz_at(rc[:, 0], rc[:, 1])
        lat = numpy.rad2deg(numpy.arcsin(numpy.clip(v[:, 2], -1, 1)))
        lon = numpy.rad2deg(numpy.arctan2(v[:, 1], v[:, 0]))
        return numpy.column_stack((lat, lon))

    def _solve(self, target, rows, cols, scans):
        """Gauss-Newton for the ``(row, col)`` within ``scans`` closest to
        ``target`` (unit vectors), from ``rows`` / ``cols``. Returns rows,
        cols, the miss and one pixel's size, both as unit-vector distances."""
        if self._scan_rows:
            low, high = scans * self._scan_rows, (scans + 1) * self._scan_rows
        else:
            low, high = numpy.zeros(len(rows)), numpy.full(len(rows), self._rows[-1] + 0.5)
        col_high = self._cols[-1] + 0.5
        for _ in range(8):
            here = self._xyz_at(rows, cols, scans)
            d_r = self._xyz_at(rows + 0.5, cols, scans) - self._xyz_at(rows - 0.5, cols, scans)
            d_c = self._xyz_at(rows, cols + 0.5, scans) - self._xyz_at(rows, cols - 0.5, scans)
            residual = target - here
            # the 3 x 2 least squares' 2 x 2 normal equations, per point
            a, b, d = (d_r * d_r).sum(1), (d_r * d_c).sum(1), (d_c * d_c).sum(1)
            e, f = (d_r * residual).sum(1), (d_c * residual).sum(1)
            det = a * d - b * b
            rows = numpy.clip(rows + (d * e - b * f) / det, low, high)
            cols = numpy.clip(cols + (a * f - b * e) / det, 0.0, col_high)
        miss = numpy.linalg.norm(target - self._xyz_at(rows, cols, scans), axis=1)
        pixel = numpy.maximum(numpy.linalg.norm(d_r, axis=1), numpy.linalg.norm(d_c, axis=1))
        return rows, cols, miss, pixel

    def latlon_to_rowcol(self, coords: Coords) -> numpy.ndarray:
        ll = _as_n2(coords)
        target = _unit_vectors(ll[:, 0], ll[:, 1])
        nodes = numpy.nan_to_num(self._xyz.reshape(-1, 3), nan=0.0)
        out = numpy.full((len(ll), 2), numpy.nan)
        for start in range(0, len(ll), 4096):
            t = target[start:start + 4096]
            nearest = (t @ nodes.T).argmax(axis=1)
            r0 = self._rows[nearest // len(self._cols)]
            c0 = self._cols[nearest % len(self._cols)]
            best = None
            # the nearest node's scan and its neighbours (overlapping scans)
            for offset in ((0,) if not self._scan_rows else (0, -1, 1)):
                scans = numpy.clip(self._scan(r0) + offset, 0, len(self._first) - 1)
                r_start = r0 if offset == 0 else (scans + 0.5) * self._scan_rows
                r, c, miss, pixel = self._solve(t, r_start, c0, scans)
                if best is None:
                    best = [r, c, miss, pixel]
                else:
                    better = miss < best[2]
                    for k, value in enumerate((r, c, miss, pixel)):
                        best[k] = numpy.where(better, value, best[k])
            r, c, miss, pixel = best
            off = ~(miss <= pixel)  # NaN counts as off
            r[off], c[off] = numpy.nan, numpy.nan
            out[start:start + 4096] = numpy.column_stack((r, c))
        return out
