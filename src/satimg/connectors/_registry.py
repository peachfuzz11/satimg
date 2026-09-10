"""Connector lookup by STAC collection name -- re-exported from the package."""

from __future__ import annotations

from satimg.connectors.base import Connector
from satimg.connectors.landsat import LandsatConnector
from satimg.connectors.sentinel import Sentinel1Connector, Sentinel2Connector

_CONNECTORS: dict[str, type[Connector]] = {
    "sentinel-1-grd": Sentinel1Connector,
    "sentinel-2-l1c": Sentinel2Connector,
    "landsat-c2l1": LandsatConnector,
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
