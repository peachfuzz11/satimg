"""ICEYE open data: the static STAC catalogue on AWS, files from the public
``iceye-open-data-catalog`` bucket. Anonymous, no account::

    connector = get_connector("iceye-grd")
    items = connector.search(geojson=aoi, start_datetime=t0, stop_datetime=t1)
    connector.download(items[0], "/data/image.zip")

``search`` reads the ``iceye-sar`` collection, which links every image
directly (the root's by-mode and by-stack views list the same images
again); there are no date catalogues to prune, so every search reads all
of its items -- several hundred small files, a few seconds.
"""

from __future__ import annotations

import logging
import posixpath

from satimg.connectors.connector import stream_to_zip
from satimg.connectors.static_stac_connector import StaticSTACConnector
from satimg.products.iceye.grd.iceye_grd_parsing import NAME

logger = logging.getLogger(__name__)

BUCKET = "https://iceye-open-data-catalog.s3.amazonaws.com"


class IceyeGRDConnector(StaticSTACConnector):
    CATALOG = BUCKET + "/collections/iceye-sar.json"
    collection = "iceye-grd"

    def _wanted(self, href: str) -> bool:
        """Image items only -- not the collection's footprint summary."""
        return NAME.match(posixpath.basename(href)) is not None

    def download(self, item: dict, save_path: str) -> str:
        """Zip the GRD (tens of MB to about a GB; it carries its own
        metadata) to ``save_path``, as ``<item id>/<item id>_GRD.tif``."""
        href = item["assets"]["grd-cog"]["href"]
        logger.info("downloading %s -> %s", item["id"], save_path)
        return stream_to_zip(self._get_session(), {f"{item['id']}/{posixpath.basename(href)}": href}, save_path)
