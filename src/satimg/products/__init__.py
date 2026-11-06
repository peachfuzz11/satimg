"""Concrete product readers. Importing this package registers them all."""

from satimg.products.landsat import LandsatProduct
from satimg.products.nisar import NisarGCOVProduct
from satimg.products.sentinel1 import Sentinel1Product
from satimg.products.sentinel2 import Sentinel2L1CProduct

__all__ = ["LandsatProduct", "NisarGCOVProduct", "Sentinel1Product", "Sentinel2L1CProduct"]
