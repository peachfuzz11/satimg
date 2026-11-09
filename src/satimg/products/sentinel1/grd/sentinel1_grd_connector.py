"""Sentinel-1 GRD (IW, ``GRDH``) from the Copernicus Data Space."""

from __future__ import annotations

from satimg.connectors.dataspace_connector import DataspaceConnector


class Sentinel1GRDConnector(DataspaceConnector):
    collection = "sentinel-1-grd"
    product_type = "IW_GRDH_1S"
