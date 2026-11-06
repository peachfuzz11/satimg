"""Remote archive connectors, keyed by STAC collection name.

``search`` is anonymous; ``download`` needs an account -- pass ``username=`` /
``password=`` (and ``token=`` for USGS / NASA Earthdata) or set the service's environment
variables (see :mod:`satimg.connectors._credentials`).
"""

from __future__ import annotations

from satimg.connectors._credentials import Credentials, CredentialsError
from satimg.connectors.base import Connector, STACConnector
from satimg.connectors.landsat import LandsatConnector
from satimg.connectors.modis import ModisConnector
from satimg.connectors.nisar import NisarConnector
from satimg.connectors.sentinel import OlciConnector, Sentinel1Connector, Sentinel2Connector
from satimg.connectors.viirs import ViirsDNBConnector

_CONNECTORS: dict[str, type[Connector]] = {
    "sentinel-1-grd": Sentinel1Connector,
    "sentinel-2-l1c": Sentinel2Connector,
    "landsat-c2l1": LandsatConnector,
    "modis-qkm": ModisConnector,
    "nisar-gcov": NisarConnector,
    "sentinel-3-olci-efr": OlciConnector,
    "viirs-dnb": ViirsDNBConnector,
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


__all__ = [
    "Connector",
    "STACConnector",
    "Sentinel1Connector",
    "Sentinel2Connector",
    "LandsatConnector",
    "ModisConnector",
    "NisarConnector",
    "OlciConnector",
    "ViirsDNBConnector",
    "Credentials",
    "CredentialsError",
    "get_connector",
]
