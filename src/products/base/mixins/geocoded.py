import abc

from metadata.base_transformer import BaseTransformer


class Geocoded(abc.ABC):
    def __init__(self):
        self._timestamp = None
        self._footprint = None
        super().__init__()

    @abc.abstractmethod
    def get_transformer(self, *args, **kwargs) -> BaseTransformer:
        pass

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def footprint(self):
        return self._footprint

    def get_bounds(self):
        lons, lats = zip(*self._footprint["geometry"]["coordinates"])
        return max(lons), min(lons), max(lats), min(lats)
