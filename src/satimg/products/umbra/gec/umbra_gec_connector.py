"""Umbra open data: the static STAC catalogue on AWS, files from the public
``umbra-open-data-catalog`` bucket. Anonymous, no account::

    connector = get_connector("umbra-gec")
    items = connector.search(geojson=aoi, start_datetime=t0, stop_datetime=t1)
    connector.download(items[0], "/data/collect.zip")

Umbra's catalogue items leave their asset ``href``\\ s empty; the files sit
under ``sar-data/task-data/<umbra:task_id>/<start time>_<satellite>/``.
``download`` lists that task's folder and takes the collect from the item's
satellite nearest the item's time (a task can hold several collects), then
zips its ``_GEC.tif`` and metadata JSON -- the CPHD / SICD / SIDD / CSI
files next to them are left behind.
"""

from __future__ import annotations

import datetime
import logging
import posixpath
import xml.etree.ElementTree as ET
from urllib.parse import quote

from satimg.connectors.connector_utils import stream_to_zip
from satimg.connectors.static_stac_connector import StaticSTACConnector
from satimg.connectors.static_stac_utils import parse_datetime
from satimg.products.umbra.gec import umbra_gec_parsing

logger = logging.getLogger(__name__)

BUCKET = "https://umbra-open-data-catalog.s3.amazonaws.com"

#: a collect folder further than this from the item's time is not its collect
MAX_OFFSET = datetime.timedelta(minutes=2)

_S3 = "{http://s3.amazonaws.com/doc/2006-03-01/}"

#: the files a GEC product needs (by name suffix)
_SUFFIXES = ("_GEC.tif", ".stac.v2.json", "_METADATA.json")


class UmbraGECConnector(StaticSTACConnector):
    CATALOG = BUCKET + "/stac/catalog.json"
    collection = "umbra-gec"

    def _list_keys(self, session, prefix: str) -> list[str]:
        """Every object key under ``prefix`` in the bucket (anonymous S3
        ``ListObjectsV2``, page by page)."""
        keys, token = [], None
        while True:
            params = {"list-type": "2", "prefix": prefix}
            if token:
                params["continuation-token"] = token
            response = session.get(BUCKET + "/", params=params, timeout=60)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            keys += [e.text for e in root.iter(f"{_S3}Key")]
            token = root.findtext(f"{_S3}NextContinuationToken")
            if root.findtext(f"{_S3}IsTruncated") != "true" or not token:
                return keys

    def collect_files(self, session, item: dict) -> tuple[str, list[str]]:
        """``(collect folder name, keys of the files to fetch)`` for ``item``."""
        props = item["properties"]
        prefix = f"sar-data/task-data/{props['umbra:task_id']}/"
        platform = props["platform"].upper()
        t = parse_datetime(props["datetime"])
        folders: dict[str, list[str]] = {}
        for key in self._list_keys(session, prefix):
            folder = key[len(prefix):].split("/", 1)[0]
            m = umbra_gec_parsing.NAME.match(folder)
            if m and m["platform"] == platform and key.endswith(_SUFFIXES):
                folders.setdefault(folder, []).append(key)
        if not folders:
            raise FileNotFoundError(f"no {platform} GEC collect under s3://umbra-open-data-catalog/{prefix}")
        folder = min(folders, key=lambda f: abs(umbra_gec_parsing.folder_time(f) - t))
        if abs(umbra_gec_parsing.folder_time(folder) - t) > MAX_OFFSET:
            raise FileNotFoundError(f"no collect near {props['datetime']} under {prefix}: {sorted(folders)}")
        keys = folders[folder]
        if not any(k.endswith("_GEC.tif") for k in keys):
            raise FileNotFoundError(f"no GEC image in {prefix}{folder}/")
        return folder, keys

    def download(self, item: dict, save_path: str) -> str:
        """Zip the collect's ``_GEC.tif`` (hundreds of MB to a GB or so) and
        metadata JSON to ``save_path``, as ``<collect folder>/<file>``."""
        session = self._get_session()
        folder, keys = self.collect_files(session, item)
        logger.info("downloading %s (%s) -> %s", item["id"], folder, save_path)
        members = {f"{folder}/{posixpath.basename(k)}": f"{BUCKET}/{quote(k)}" for k in keys}
        return stream_to_zip(session, members, save_path)
