import abc
import datetime

from satproducts.transformers.base_transformer import BaseTransformer


class GeoMixin(abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    @abc.abstractmethod
    def transformer(self, *args, **kwargs) -> BaseTransformer:
        pass

    @property
    @abc.abstractmethod
    def timestamp(self) -> datetime.datetime:
        pass

    @property
    @abc.abstractmethod
    def footprint(self) -> dict:
        pass

    def get_bounds(self):
        lons, lats = zip(*self.footprint["geometry"]["coordinates"])
        return max(lons), min(lons), max(lats), min(lats)
