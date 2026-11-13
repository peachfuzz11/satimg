"""satimg -- a uniform, tile-friendly wrapper over satellite imagery.

    import satimg

    product = satimg.open("/data/S2A_MSIL1C_....SAFE")

    for patch in product.patches(512, overlap=64):
        do_something(patch.raw.values)    # (band, y, x) native dtype
        print(patch.col, patch.row)       # where it sits in the full image
        print(patch.meta.sample())        # {field: value} at the patch centre
"""

import logging
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

# Log under the ``satimg.*`` namespace; the app attaches handlers.
logging.getLogger(__name__).addHandler(logging.NullHandler())

try:
    __version__ = _version("satimg")
except PackageNotFoundError:  # a source tree without an install
    __version__ = "0.0.0"

from satimg.connectors._credentials import CredentialsError
from satimg.connectors.connector import Connector
from satimg.connectors.connector_factory import get_connector
from satimg.geometry import Grid, Window, windows_at
from satimg.metadata.field import Field
from satimg.metadata.metadata import Metadata
from satimg.patch import Patch
from satimg.prediction.characteriser import Characteriser
from satimg.prediction.detector import Detector, DetectorConfig
from satimg.prediction.models import CharacterisationModel, Yolo26Model
from satimg.prediction.types import BBox, Characterisation, Coordinate, Detection
from satimg.prediction.viirs_boat_detector import ViirsBoatConfig, ViirsBoatDetector
from satimg.product import Product, open, open_zip
from satimg.products.capella.gec.capella_gec import CapellaGEC
from satimg.products.capella.gec.capella_gec_connector import CapellaGECConnector
from satimg.products.iceye.grd.iceye_grd import IceyeGRD
from satimg.products.iceye.grd.iceye_grd_connector import IceyeGRDConnector
from satimg.products.landsat.c2l1.landsat_c2l1 import LandsatC2L1
from satimg.products.landsat.c2l1.landsat_c2l1_connector import LandsatC2L1Connector
from satimg.products.modis.qkm.modis_qkm import ModisQKM
from satimg.products.modis.qkm.modis_qkm_connector import ModisQKMConnector
from satimg.products.nisar.gcov.nisar_gcov import NisarGCOV
from satimg.products.nisar.gcov.nisar_gcov_connector import NisarGCOVConnector
from satimg.products.sentinel1.grd.sentinel1_grd import Sentinel1GRD
from satimg.products.sentinel1.grd.sentinel1_grd_connector import Sentinel1GRDConnector
from satimg.products.sentinel2.l1c.sentinel2_l1c import Sentinel2L1C
from satimg.products.sentinel2.l1c.sentinel2_l1c_connector import Sentinel2L1CConnector
from satimg.products.sentinel3.olci.sentinel3_olci import Sentinel3OLCI
from satimg.products.sentinel3.olci.sentinel3_olci_connector import (
    Sentinel3OLCIConnector,
)
from satimg.products.umbra.gec.umbra_gec import UmbraGEC
from satimg.products.umbra.gec.umbra_gec_connector import UmbraGECConnector
from satimg.products.viirs.dnb.viirs_dnb import ViirsDNB
from satimg.products.viirs.dnb.viirs_dnb_connector import ViirsDNBConnector
from satimg.registry import UnknownProductError, scene_key
from satimg.sar_orbit_geometry import SAROrbitGeometry
from satimg.sar_product import SARProduct
from satimg.scene_geometry import SceneGeometry
from satimg.tiling import as_band_yx, label_bands, read_window

__all__ = [
    "__version__",
    # opening products
    "open",
    "open_zip",
    "Product",
    "SARProduct",
    "Patch",
    "UnknownProductError",
    "scene_key",
    "CapellaGEC",
    "IceyeGRD",
    "LandsatC2L1",
    "ModisQKM",
    "NisarGCOV",
    "Sentinel1GRD",
    "Sentinel2L1C",
    "Sentinel3OLCI",
    "UmbraGEC",
    "ViirsDNB",
    # tiling
    "Window",
    "Grid",
    "windows_at",
    "read_window",
    "as_band_yx",
    "label_bands",
    # geometry and metadata
    "SceneGeometry",
    "SAROrbitGeometry",
    "Metadata",
    "Field",
    # downloading
    "get_connector",
    "Connector",
    "CredentialsError",
    "CapellaGECConnector",
    "IceyeGRDConnector",
    "LandsatC2L1Connector",
    "ModisQKMConnector",
    "NisarGCOVConnector",
    "Sentinel1GRDConnector",
    "Sentinel2L1CConnector",
    "Sentinel3OLCIConnector",
    "UmbraGECConnector",
    "ViirsDNBConnector",
    # prediction
    "Detector",
    "DetectorConfig",
    "Yolo26Model",
    "Characteriser",
    "CharacterisationModel",
    "ViirsBoatDetector",
    "ViirsBoatConfig",
    "Detection",
    "Characterisation",
    "BBox",
    "Coordinate",
]
