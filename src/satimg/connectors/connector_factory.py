"""Picking the :class:`~satimg.connectors.connector.Connector` for a STAC collection.

``search`` is anonymous; ``download`` needs an account -- pass ``username=`` /
``password=`` (and ``token=`` for USGS / NASA Earthdata) or set the service's
environment variables (see :mod:`satimg.connectors._credentials`) -- except
for Umbra and Capella, whose open data downloads anonymously too.
"""

from __future__ import annotations

from satimg.connectors.connector import Connector
from satimg.products.capella.gec.capella_gec_connector import CapellaGECConnector
from satimg.products.landsat.c2l1.landsat_c2l1_connector import LandsatC2L1Connector
from satimg.products.modis.qkm.modis_qkm_connector import ModisQKMConnector
from satimg.products.nisar.gcov.nisar_gcov_connector import NisarGCOVConnector
from satimg.products.sentinel1.grd.sentinel1_grd_connector import Sentinel1GRDConnector
from satimg.products.sentinel2.l1c.sentinel2_l1c_connector import Sentinel2L1CConnector
from satimg.products.sentinel3.olci.sentinel3_olci_connector import (
    Sentinel3OLCIConnector,
)
from satimg.products.umbra.gec.umbra_gec_connector import UmbraGECConnector
from satimg.products.viirs.dnb.viirs_dnb_connector import ViirsDNBConnector

_CONNECTORS: dict[str, type[Connector]] = {
    "sentinel-1-grd": Sentinel1GRDConnector,
    "sentinel-2-l1c": Sentinel2L1CConnector,
    "landsat-c2l1": LandsatC2L1Connector,
    "modis-qkm": ModisQKMConnector,
    "nisar-gcov": NisarGCOVConnector,
    "sentinel-3-olci-efr": Sentinel3OLCIConnector,
    "viirs-dnb": ViirsDNBConnector,
    "umbra-gec": UmbraGECConnector,
    "capella-gec": CapellaGECConnector,
}


def get_connector(collection: str, **credentials) -> Connector:
    """Instantiate the connector for a STAC ``collection`` name.

    Extra keyword arguments (``username``, ``password``, ``token``) are passed to
    the connector's constructor.
    """
    try:
        cls = _CONNECTORS[collection]
    except KeyError:
        raise ValueError(
            f"no connector for {collection!r}; known: {sorted(_CONNECTORS)}"
        ) from None
    return cls(**credentials)
