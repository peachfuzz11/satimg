from typing import Union, Tuple, List

import numpy
import rasterio

from metadata.transformer import Transformer


class GCPTransformer(Transformer):

    def __init__(self, gcps, crs):
        transform = rasterio.transform.GCPTransformer(gcps)
        super().__init__(transform, crs)

    def latlon_to_rowcol(self,
                         coords: Union[Tuple[float, float], List[Tuple[float, float]], numpy.ndarray]) -> numpy.ndarray:
        """
        Converts latitude and longitude coordinates to row and column indices.

        Parameters:
            coords (Union[Tuple[float, float], List[Tuple[float, float]], numpy.ndarray]):
                A single (lat, lon) tuple, a list of such tuples, or a numpy array of shape (N, 2).

        Returns:
            numpy.ndarray: An (N, 2) array of (row, col) index values.
        """
        coords = self._to_numpy(coords)
        rows, cols = self._transform.rowcol(coords[:, 1], coords[:, 0])
        return numpy.column_stack((rows, cols))

    def rowcol_to_latlon(self, coords: Union[Tuple[int, int], List[Tuple[int, int]], numpy.ndarray]) -> numpy.ndarray:
        """
        Converts row and column indices to latitude and longitude coordinates.

        Parameters:
            coords (Union[Tuple[int, int], List[Tuple[int, int]], numpy.ndarray]):
                A single (row, col) tuple, a list of such tuples, or a numpy array of shape (N, 2).

        Returns:
            numpy.ndarray: An (N, 2) array of (lat, lon) coordinate values.
        """
        coords = self._to_numpy(coords)
        lons, lats = self._transform.xy(coords[:, 0], coords[:, 1], offset="center")
        return numpy.column_stack((lats, lons))
