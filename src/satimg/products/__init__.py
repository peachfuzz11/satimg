"""Concrete product readers. Importing this package registers them all."""

from satimg.products.landsat.product import LandsatProduct
from satimg.products.modis.product import ModisProduct
from satimg.products.nisar.product import NisarGCOVProduct
from satimg.products.olci.product import OlciEFRProduct
from satimg.products.sentinel1.product import Sentinel1Product
from satimg.products.sentinel2.product import Sentinel2L1CProduct
from satimg.products.viirs.product import ViirsDNBProduct

__all__ = ["LandsatProduct", "ModisProduct", "NisarGCOVProduct", "OlciEFRProduct", "Sentinel1Product", "Sentinel2L1CProduct", "ViirsDNBProduct"]
