"""A connector over a static STAC catalogue: a tree of ``catalog.json`` /
item files in a bucket, with no search API behind it (Umbra's, Capella's and
ICEYE's open data).

``search`` walks the tree from :attr:`StaticSTACConnector.CATALOG` down,
level by level, fetching each level's catalogues in parallel. A child
catalogue whose title or directory names a date period -- ``2025``,
``2025-08``, ``2025-08-13``, as both buckets lay theirs out -- is skipped
when that period falls outside the search's time range, so a narrow date
range reads only a handful of catalogues. Every item left is fetched and
filtered on its ``datetime`` and geometry here. Without a time range the
whole catalogue is read: a few thousand small files, some seconds.

Everything is anonymous; nothing needs an account.
"""

from __future__ import annotations

import abc
import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin

import requests
import shapely.geometry

from satimg.connectors.connector import Connector
from satimg.connectors.static_stac_utils import link_period, overlaps, parse_datetime, utc
from satimg.geojson import geometry_of

logger = logging.getLogger(__name__)

#: parallel requests while walking a catalogue
WORKERS = 16


class StaticSTACConnector(Connector, abc.ABC):
    """A :class:`Connector` over a static STAC catalogue rooted at
    :attr:`CATALOG`. A subclass sets :attr:`CATALOG`, may narrow the items
    with :meth:`_wanted`, and implements ``download``."""

    #: URL of the root ``catalog.json``
    CATALOG: str

    def _get_session(self) -> requests.Session:
        """An anonymous session -- the buckets are public."""
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_maxsize=WORKERS)
        session.mount("https://", adapter)
        return session

    def _wanted(self, href: str) -> bool:
        """Whether to fetch the item at ``href`` -- every item by default; a
        catalogue that lists several product types side by side (Capella's)
        picks its own by file name here, before any item is fetched."""
        return True

    def _fetch_all(self, session: requests.Session, urls: list[str]) -> list[dict]:
        def fetch(url: str) -> dict:
            response = session.get(url, timeout=60)
            response.raise_for_status()
            return response.json()

        with ThreadPoolExecutor(WORKERS) as pool:
            return list(pool.map(fetch, urls))

    def _item_hrefs(self, session, start, stop) -> list[str]:
        """Every wanted item URL under :attr:`CATALOG` whose date catalogues
        overlap ``[start, stop]``."""
        items, level = [], [self.CATALOG]
        while level:
            children = []
            for url, doc in zip(level, self._fetch_all(session, level)):
                for link in doc.get("links", []):
                    href = urljoin(url, link["href"])
                    if link["rel"] == "item" and self._wanted(href):
                        items.append(href)
                    elif link["rel"] == "child" and overlaps(link_period(link), start, stop):
                        children.append(href)
            logger.debug("static STAC: %d catalogue(s) read, %d item(s) so far", len(level), len(items))
            level = children
        return items

    def search(
        self,
        geojson: dict | None = None,
        start_datetime: datetime.datetime | None = None,
        stop_datetime: datetime.datetime | None = None,
        ids: list[str] | None = None,
        exclude_ids: list[str] | None = None,
        **kwargs,
    ) -> list[dict]:
        """STAC items (dicts) whose ``datetime`` lies in ``[start_datetime,
        stop_datetime]`` (naive datetimes are UTC) and whose geometry
        intersects ``geojson``, oldest first."""
        start, stop = utc(start_datetime), utc(stop_datetime)
        session = self._get_session()
        hrefs = self._item_hrefs(session, start, stop)
        area = shapely.geometry.shape(geometry_of(geojson)) if geojson is not None else None
        items = []
        for item in self._fetch_all(session, hrefs):
            t = parse_datetime(item["properties"]["datetime"])
            if (start and t < start) or (stop and t > stop):
                continue
            if area is not None and not area.intersects(shapely.geometry.shape(item["geometry"])):
                continue
            if (ids and item["id"] not in ids) or (exclude_ids and item["id"] in exclude_ids):
                continue
            items.append(item)
        items.sort(key=lambda i: i["properties"]["datetime"])
        logger.debug("static STAC search: %d of %d item(s) match", len(items), len(hrefs))
        return items
