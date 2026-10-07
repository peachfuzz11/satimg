"""Pixel <-> geographic coordinate conversion.

:class:`Transformer` wraps an affine ``transform`` + ``crs`` (the common case).
:class:`GCPTransformer` does the same job for products that only ship ground
control points (Sentinel-1).

Both accept a single ``(a, b)`` pair, a list of pairs, or an ``(N, 2)`` array and
always return an ``(N, 2)`` array. ``rowcol`` is ``(row, col)`` = ``(y, x)``;
``latlon`` is ``(lat, lon)``.

Both round-trip through a JSON-serialisable dict (:meth:`Transformer.to_dict` /
:meth:`Transformer.from_dict`), so a product's georeferencing can be stored and
used after the product file is gone.
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
        return {"type": "affine", "transform": list(self._transform)[:6], "crs": self._crs.to_wkt()}

    @staticmethod
    def from_dict(data: dict) -> Transformer:
        """Rebuild a :class:`Transformer` or :class:`GCPTransformer` from
        :meth:`to_dict`."""
        crs = CRS.from_wkt(data["crs"])
        if data["type"] == "gcp":
            gcps = [GroundControlPoint(row=r, col=c, x=x, y=y, z=z) for r, c, x, y, z in data["gcps"]]
            return GCPTransformer(gcps, crs)
        return Transformer(Affine(*data["transform"]), crs)

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


class GCPTransformer(Transformer):
    """Coordinate transform backed by ground control points."""

    def __init__(self, gcps, crs):
        super().__init__(rasterio.transform.GCPTransformer(gcps), crs)
        self._gcps = list(gcps)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(crs={self._crs}, gcps={len(self._gcps)})"

    def to_dict(self) -> dict:
        return {
            "type": "gcp",
            "gcps": [[g.row, g.col, g.x, g.y, g.z] for g in self._gcps],
            "crs": self._crs.to_wkt(),
        }

    def rowcol_to_latlon(self, coords: Coords) -> numpy.ndarray:
        rc = _as_n2(coords)
        lons, lats = self._transform.xy(rc[:, 0].tolist(), rc[:, 1].tolist(), offset="center")
        return numpy.column_stack((numpy.asarray(lats), numpy.asarray(lons)))

    def latlon_to_rowcol(self, coords: Coords) -> numpy.ndarray:
        ll = _as_n2(coords)
        rows, cols = self._transform.rowcol(ll[:, 1].tolist(), ll[:, 0].tolist())
        return numpy.column_stack((numpy.asarray(rows), numpy.asarray(cols)))
