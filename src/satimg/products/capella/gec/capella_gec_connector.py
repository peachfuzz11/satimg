"""Capella open data: the static STAC catalogue on AWS, files from the public
``capella-open-data`` bucket. Anonymous, no account::

    connector = get_connector("capella-gec")
    items = connector.search(geojson=aoi, start_datetime=t0, stop_datetime=t1)
    connector.download(items[0], "/data/collect.zip")

``search`` walks the by-datetime catalogue, which lists every product type
of a collect side by side; only the GEC items are fetched.
"""

from __future__ import annotations

import logging
import posixpath

from satimg.connectors.connector_utils import stream_to_zip
from satimg.connectors.static_stac_connector import StaticSTACConnector

logger = logging.getLogger(__name__)

BUCKET = "https://capella-open-data.s3.amazonaws.com"


class CapellaGECConnector(StaticSTACConnector):
    CATALOG = BUCKET + "/stac/capella-open-data-by-datetime/catalog.json"
    collection = "capella-gec"

    def _wanted(self, href: str) -> bool:
        return "_GEC_" in posixpath.basename(href)

    def download(self, item: dict, save_path: str) -> str:
        """Zip the GEC image (a few hundred MB), its extended metadata and
        the quick-look to ``save_path``, as ``<item id>/<file>``."""
        assets = item["assets"]
        image = next(
            a for a in assets.values() if "data" in a.get("roles", []) and a["href"].endswith(".tif")
        )
        hrefs = [image["href"], assets["metadata"]["href"]]
        if "thumbnail" in assets:
            hrefs.append(assets["thumbnail"]["href"])
        logger.info("downloading %s -> %s", item["id"], save_path)
        members = {f"{item['id']}/{posixpath.basename(h)}": h for h in hrefs}
        return stream_to_zip(self._get_session(), members, save_path)
