"""Sentinel-2 L1C from the Copernicus Data Space."""

from __future__ import annotations

from satimg.connectors.copernicus_dataspace_connector import (
    CopernicusDataspaceConnector,
)


class Sentinel2L1CConnector(CopernicusDataspaceConnector):
    collection = "sentinel-2-l1c"
