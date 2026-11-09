"""Georeferencing of a sensor-geometry swath by its per-pixel geolocation."""

from __future__ import annotations

import numpy

from satimg.transform.transformer import Coords, Transformer, as_n2

#: mean earth radius, metres
EARTH_RADIUS_M = 6_371_000.0


def unit_vectors(lat, lon) -> numpy.ndarray:
    """``(..., 3)`` earth-centred unit vectors of ``lat`` / ``lon`` degrees."""
    lat, lon = numpy.deg2rad(lat), numpy.deg2rad(lon)
    return numpy.stack([numpy.cos(lat) * numpy.cos(lon), numpy.cos(lat) * numpy.sin(lon), numpy.sin(lat)], axis=-1)


class SwathTransformer(Transformer):
    """Coordinate conversion for a sensor-geometry swath (VIIRS, MODIS, OLCI)
    that ships a latitude / longitude for every pixel: those arrays kept on
    a coarse ``rows`` x ``cols`` node grid (pixel coordinates, pixel centres
    at ``i + 0.5``) and interpolated -- bilinearly, on the unit sphere, so
    the antimeridian and the poles need no special casing.

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
        self._xyz = unit_vectors(self._lat, self._lon)
        self._scan_rows = scan_rows
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
        rc = as_n2(coords)
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
        ll = as_n2(coords)
        target = unit_vectors(ll[:, 0], ll[:, 1])
        nodes = numpy.nan_to_num(self._xyz.reshape(-1, 3), nan=0.0)
        out = numpy.full((len(ll), 2), numpy.nan)
        # the nearest-node search is a (points x nodes) product: bound it to
        # ~20M entries, whatever the node grid's size
        chunk = int(max(64, min(4096, 20_000_000 // max(len(nodes), 1))))
        for start in range(0, len(ll), chunk):
            t = target[start:start + chunk]
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
            out[start:start + chunk] = numpy.column_stack((r, c))
        return out
