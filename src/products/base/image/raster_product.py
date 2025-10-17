import abc
import gc
import sys
from typing import Tuple

import numpy

from common.image_slice import ImageSlice
from products.base.mixins.geocoded import Geocoded
from products.base.mixins.sliceable import Sliceable
from products.base.mixins.viewable import Viewable
from products.base.product import Product


class RasterProduct(Product, Sliceable, Viewable, Geocoded, abc.ABC):
    def __init__(self, product_path):
        self._product_path = product_path
        self._product = None
        self.metadata = None
        self._width = None
        self._height = None
        self._nr_bands = None
        super().__init__()

    @property
    def product(self):
        return self._product

    def read_slice(self, image_slice: ImageSlice):
        subset = self.product.isel(**image_slice.to_slice())
        subset.attrs["i"] = image_slice.i
        subset.attrs["j"] = image_slice.j
        return subset

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def nr_bands(self):
        return self._nr_bands

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self._nr_bands, self.height, self.width) if self._nr_bands else (1, self.height, self.width)

    @property
    def product_path(self):
        return self._product_path

    @abc.abstractmethod
    def __enter__(self) -> "RasterProduct":
        pass

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        del self._product
        gc.collect()

    def __repr__(self) -> str:
        return f"RasterProduct(product_path={self._product_path})"

    def __str__(self) -> str:
        return f"Instance of a satellite product for {self._product_path}"

    def __sizeof__(self) -> tuple:
        base_size = super().__sizeof__()
        non_numpy_size = base_size
        numpy_size = 0

        attributes = [
            self._product_path,
            self.metadata,
            self._height,
            self._width,
            self._nr_bands,
        ]

        # Include size of the product (if loaded)
        if self._product is not None:
            try:
                # If it's a Dataset, iterate over data variables
                if hasattr(self._product, "data_vars"):
                    for var_name, var in self._product.data_vars.items():
                        numpy_size += var.nbytes
                # If it's a DataArray, calculate its nbytes directly
                elif hasattr(self._product, "nbytes"):
                    numpy_size += self._product.nbytes
                else:
                    numpy_size += sys.getsizeof(self._product)
            except:
                numpy_size = 0

        # Calculate size of non-numpy attributes
        for attr in attributes:
            if isinstance(attr, numpy.ndarray):
                numpy_size += attr.nbytes
            elif hasattr(attr, "__sizeof__"):
                non_numpy_size += attr.__sizeof__()
            else:
                non_numpy_size += sys.getsizeof(attr)

        non_numpy_size_mb = round(non_numpy_size / (1024 ** 2), 5)
        numpy_size_mb = round(numpy_size / (1024 ** 2), 2)

        return non_numpy_size_mb, numpy_size_mb

    @property
    def size(self):
        return self.__sizeof__()
