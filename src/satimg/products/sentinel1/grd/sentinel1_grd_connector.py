"""Sentinel-1 GRD (IW, ``GRDH``) from the Copernicus Data Space."""

from __future__ import annotations

from satimg.connectors.copernicus_dataspace_connector import CopernicusDataspaceConnector


class Sentinel1GRDConnector(CopernicusDataspaceConnector):
    collection = "sentinel-1-grd"
    product_type = "IW_GRDH_1S"
