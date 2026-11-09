"""Sentinel-2 L1C from the Copernicus Data Space."""

from __future__ import annotations

from satimg.connectors.dataspace_connector import DataspaceConnector


class Sentinel2L1CConnector(DataspaceConnector):
    collection = "sentinel-2-l1c"
