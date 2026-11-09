"""Landsat Collection 2 Level-1 from USGS LandsatLook (a USGS ERS account,
see :mod:`satimg.connectors.usgs_ers_connector`)::

    LandsatC2L1Connector(username="me", password="...", token="...")
"""

from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path

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
        session = self._get_session()
        # stored: the rasters are already compressed, and GDAL reads stored members
        # in place at file speed (see satimg.source.zip_source.ZipSource)
        with zipfile.ZipFile(save_path, "w", zipfile.ZIP_STORED) as archive:
            for name, asset in item["assets"].items():
                if name not in _ASSETS:
                    continue
                href = asset["href"]
                filename = name if Path(name).suffix else name + Path(href).suffix
                logger.debug("fetching asset %s: %s", name, href)
                response = session.get(href, timeout=120)
                response.raise_for_status()
                archive.writestr(os.path.join(item["id"], filename), response.content)
            if archive.testzip() is not None:
                raise zipfile.BadZipFile("invalid zip archive")
        logger.info("wrote %s", save_path)
        return save_path
