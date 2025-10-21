from typing import Union, Tuple, List

import numpy
from rasterio import warp

from satproducts.transformers.base_transformer import BaseTransformer


class Transformer(BaseTransformer):

    def __init__(self, transform, crs):
        super().__init__(transform, crs)

    def latlon_to_rowcol(self,
                         coords: Union[Tuple[float, float], List[Tuple[float, float]], numpy.ndarray]) -> numpy.ndarray:
        """
        Converts latitude and longitude coordinates to row and column indices.

        Parameters:
            coords: A single (lat, lon) tuple, a list of such tuples, or a numpy array of shape (N, 2).

        Returns:
            numpy.ndarray: An (N, 2) array of (row, col) index values.
        """
        coords = self._to_numpy(coords)

        if not self._crs.is_geographic:
            lons, lats = warp.transform("EPSG:4326", self._crs, coords[:, 1].tolist(), coords[:, 0].tolist())
            coords[:, 0], coords[:, 1] = numpy.array(lats), numpy.array(lons)
            #coords[:, 1], coords[:, 0] = warp.transform("EPSG:4326", self._crs, coords[:, 1], coords[:, 0])

        cols, rows = ~self._transform * (coords[:, 1], coords[:, 0])
        return numpy.column_stack((rows, cols))

    def rowcol_to_latlon(self,
                         coords: Union[Tuple[float, float], List[Tuple[float, float]], numpy.ndarray]) -> numpy.ndarray:
        """
        Converts row and column indices to latitude and longitude coordinates.

        Parameters:
            coords: A single (row, col) tuple, a list of such tuples, or a numpy array of shape (N, 2).

        Returns:
            numpy.ndarray: An (N, 2) array of (lat, lon) coordinate values.
        """
        coords = self._to_numpy(coords)

        lons, lats = self._transform * (coords[:, 1], coords[:, 0])

        if not self._crs.is_geographic:
           lons, lats = warp.transform(self._crs, "EPSG:4326", lons.tolist(), lats.tolist())
           lons, lats = numpy.array(lons), numpy.array(lats)
           #lons, lats = warp.transform(self._crs, "EPSG:4326", lons, lats)

        return numpy.column_stack((lats, lons))

    def _to_numpy(self, coords):
        if not isinstance(coords, numpy.ndarray):
            coords = numpy.array(coords, dtype=float)
        else:
            coords = coords.copy()
        if coords.ndim == 1:
            coords = coords.reshape(1, 2)
        return coords
