"""Georeferencing by ground control points (Sentinel-1)."""

from __future__ import annotations

import numpy
import rasterio.transform
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS

from satimg.transform.base import Coords, Transformer, as_n2


class GCPTransformer(Transformer):
    """Coordinate conversion backed by ground control points, for products
    that ship nothing else (Sentinel-1 GRD). ``latlon_to_rowcol`` snaps to
    whole pixels."""

    key = "gcps"

    def __init__(self, gcps, crs: CRS):
        self._gcps = list(gcps)
        self._crs = crs
        self._transform = rasterio.transform.GCPTransformer(self._gcps)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(crs={self._crs}, gcps={len(self._gcps)})"

    def to_dict(self) -> dict:
        return {
            "gcps": [[g.row, g.col, g.x, g.y, g.z] for g in self._gcps],
            "crs": self._crs.to_wkt(),
        }

    @classmethod
    def _from_dict(cls, data: dict) -> GCPTransformer:
        gcps = [GroundControlPoint(row=r, col=c, x=x, y=y, z=z) for r, c, x, y, z in data["gcps"]]
        return cls(gcps, CRS.from_wkt(data["crs"]))

    def rowcol_to_latlon(self, coords: Coords) -> numpy.ndarray:
        rc = as_n2(coords)
        lons, lats = self._transform.xy(rc[:, 0].tolist(), rc[:, 1].tolist(), offset="center")
        return numpy.column_stack((numpy.asarray(lats), numpy.asarray(lons)))

    def latlon_to_rowcol(self, coords: Coords) -> numpy.ndarray:
        ll = as_n2(coords)
        rows, cols = self._transform.rowcol(ll[:, 1].tolist(), ll[:, 0].tolist())
        return numpy.column_stack((numpy.asarray(rows), numpy.asarray(cols)))
