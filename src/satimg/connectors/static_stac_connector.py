"""A connector over a static STAC catalogue: a tree of ``catalog.json`` /
item files in a bucket, with no search API behind it (Umbra's and Capella's
open data).

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
import posixpath
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin

import requests
import shapely.geometry

from satimg.connectors.connector import Connector
from satimg.geojson import geometry_of

logger = logging.getLogger(__name__)

#: parallel requests while walking a catalogue
WORKERS = 16

#: a trailing ``YYYY``, ``YYYY-MM`` or ``YYYY-MM-DD``
_PERIOD = re.compile(r"(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?$")


def _utc(t: datetime.datetime | None) -> datetime.datetime | None:
    """``t`` timezone-aware; a naive datetime is taken as UTC."""
    if t is None or t.tzinfo is not None:
        return t
    return t.replace(tzinfo=datetime.timezone.utc)


def parse_datetime(text: str) -> datetime.datetime:
    """A STAC ``datetime`` (``...Z`` or ``+00:00``) as an aware datetime."""
    return datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))


def link_period(link: dict) -> tuple[datetime.datetime, datetime.datetime] | None:
    """``[start, end)`` of the date period a child link names in its title
    (``"2025-08"``, ``"Capella Open Data 2025-08"``) or its directory
    (``./2025-08/catalog.json``) -- ``None`` if it names none."""
    directory = posixpath.basename(posixpath.dirname(link["href"]))
    for text in (link.get("title") or "", directory):
        m = _PERIOD.search(text.strip())
        if not m:
            continue
        year, month, day = (int(g) if g else None for g in m.groups())
        utc = datetime.timezone.utc
        if month is None:
            return datetime.datetime(year, 1, 1, tzinfo=utc), datetime.datetime(year + 1, 1, 1, tzinfo=utc)
        if day is None:
            end = datetime.datetime(year + month // 12, month % 12 + 1, 1, tzinfo=utc)
            return datetime.datetime(year, month, 1, tzinfo=utc), end
        start = datetime.datetime(year, month, day, tzinfo=utc)
        return start, start + datetime.timedelta(days=1)
    return None


def _overlaps(period, start, stop) -> bool:
    if period is None:
        return True
    return (start is None or period[1] > start) and (stop is None or period[0] <= stop)


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
                    elif link["rel"] == "child" and _overlaps(link_period(link), start, stop):
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
        start, stop = _utc(start_datetime), _utc(stop_datetime)
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
