import abc

from products.base.mixins.geo_mixin import GeoMixin
from products.base.mixins.slice_mixin import SliceMixin
from products.base.mixins.view_mixin import ViewMixin


class Product(ViewMixin, SliceMixin, GeoMixin, abc.ABC):
    def __init__(self, product_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._product_path = product_path

    @property
    def product_path(self):
        return self._product_path
