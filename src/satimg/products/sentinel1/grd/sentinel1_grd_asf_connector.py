"""Sentinel-1 GRD (IW, ``GRDH``) from the Alaska Satellite Facility.

The ASF archive holds both Sentinel-1A and Sentinel-1B -- ``GRD_HD`` is the
original ESA product, under its original name (checksum included), which the
Copernicus Data Space STAC catalogue (:class:`Sentinel1GRDConnector`) does not:
that one only lists the COG conversions of Sentinel-1A, Sentinel-1C and later.

``search`` is anonymous; ``download`` needs a free NASA Earthdata Login (see
:mod:`satimg.connectors.asf_connector`) and writes the product's own zip,
which ``satimg.open`` reads as it is::

    connector = get_connector("sentinel-1-grd-asf")
    items = connector.search(ids=["S1B_IW_GRDH_1SDV_20200407T185759_20200407T185824_021042_027EAA_107A"])
    connector.download(items[0], "/data/scene.zip")
"""

from __future__ import annotations

from satimg.connectors.asf_connector import ASFConnector


class Sentinel1GRDASFConnector(ASFConnector):
    """IW ``GRD_HD`` products of Sentinel-1A and Sentinel-1B. Pass
    ``platform="S1B"`` (an ASF search parameter) to ``search`` for one of them."""

    collection = "sentinel-1-grd-asf"
    DATASET = "SENTINEL-1"
    PROCESSING_LEVEL = "GRD_HD"
    SEARCH_DEFAULTS = {"beamMode": "IW"}
