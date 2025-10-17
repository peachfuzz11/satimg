import abc
from typing import Union, Tuple, List

import numpy


class BaseTransformer(abc.ABC):

    def __init__(self, transform, crs):
        self._transform = transform
        self._crs = crs

    @abc.abstractmethod
    def latlon_to_rowcol(self,
                         coords: Union[Tuple[float, float], List[Tuple[float, float]], numpy.ndarray]) -> numpy.ndarray:
        pass

    @abc.abstractmethod
    def rowcol_to_latlon(self,
                         coords: Union[Tuple[float, float], List[Tuple[float, float]], numpy.ndarray]) -> numpy.ndarray:
        pass

    def __repr__(self):
        return f"Transformer(crs={self._crs} transform={self._transform})"
