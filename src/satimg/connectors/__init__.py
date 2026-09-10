"""Remote archive connectors, keyed by STAC collection name.

``search`` is anonymous; ``download`` needs an account -- pass ``username=`` /
``password=`` (and ``token=`` for USGS) or set the service's environment
variables (see :mod:`satimg.connectors._credentials`).
"""

from satimg.connectors._credentials import Credentials, CredentialsError
from satimg.connectors._registry import get_connector
from satimg.connectors.base import Connector, STACConnector
from satimg.connectors.landsat import LandsatConnector
from satimg.connectors.sentinel import Sentinel1Connector, Sentinel2Connector

__all__ = [
    "Connector",
    "STACConnector",
    "Sentinel1Connector",
    "Sentinel2Connector",
    "LandsatConnector",
    "Credentials",
    "CredentialsError",
    "get_connector",
]
