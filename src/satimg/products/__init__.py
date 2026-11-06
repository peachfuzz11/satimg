"""Concrete product readers. Importing this package registers them all."""

from satimg.products.landsat import LandsatProduct
from satimg.products.modis import ModisProduct
from satimg.products.nisar import NisarGCOVProduct
from satimg.products.olci import OlciEFRProduct
from satimg.products.sentinel1 import Sentinel1Product
from satimg.products.sentinel2 import Sentinel2L1CProduct
from satimg.products.viirs import ViirsDNBProduct

__all__ = ["LandsatProduct", "ModisProduct", "NisarGCOVProduct", "OlciEFRProduct", "Sentinel1Product", "Sentinel2L1CProduct", "ViirsDNBProduct"]
