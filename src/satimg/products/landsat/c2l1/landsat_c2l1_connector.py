"""Landsat Collection 2 Level-1 from USGS LandsatLook (a USGS ERS account,
see :mod:`satimg.connectors.usgs_ers_connector`)::

    LandsatC2L1Connector(username="me", password="...", token="...")
"""

from __future__ import annotations

import logging
from pathlib import Path

from satimg.connectors.connector_utils import stream_to_zip
from satimg.connectors.usgs_ers_connector import UsgsErsConnector

logger = logging.getLogger(__name__)

_ASSETS = (
    "thumbnail", "MTL.json", "coastal", "blue", "green", "red", "nir08", "swir16",
    "swir22", "pan", "cirrus", "lwir11", "lwir12", "qa_pixel", "qa_radsat",
    "ANG.txt", "VAA", "VZA", "SAA", "SZA",
)


class LandsatC2L1Connector(UsgsErsConnector):
    STAC_ENDPOINT = "https://landsatlook.usgs.gov/stac-server"
    collection = "landsat-c2l1"

    def download(self, item: dict, save_path: str) -> str:
        assets = [name for name in item["assets"] if name in _ASSETS]
        logger.info(
            "downloading %s (%d assets) -> %s", item["id"], len(assets), save_path
        )
        members = {}
        for name in assets:
            href = item["assets"][name]["href"]
            filename = name if Path(name).suffix else name + Path(href).suffix
            members[f"{item['id']}/{filename}"] = href
        return stream_to_zip(self._get_session(), members, save_path)
