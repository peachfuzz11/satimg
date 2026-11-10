import datetime

import pytest

from satimg.connectors.connector_factory import get_connector
from satimg.connectors.static_stac_connector import StaticSTACConnector, link_period

UTC = datetime.timezone.utc


@pytest.mark.parametrize("link, period", [
    ({"href": "./2025/catalog.json", "title": "2025"},
     (datetime.datetime(2025, 1, 1, tzinfo=UTC), datetime.datetime(2026, 1, 1, tzinfo=UTC))),
    ({"href": "./x/catalog.json", "title": "Capella Open Data 2025-12"},
     (datetime.datetime(2025, 12, 1, tzinfo=UTC), datetime.datetime(2026, 1, 1, tzinfo=UTC))),
    ({"href": "./capella-open-data-2026-09-09/catalog.json"},
     (datetime.datetime(2026, 9, 9, tzinfo=UTC), datetime.datetime(2026, 9, 10, tzinfo=UTC))),
    ({"href": "./by-use-case/catalog.json", "title": "By Use Case"}, None),
])
def test_link_period(link, period):
    assert link_period(link) == period


def _item(item_id, when, lon):
    return {
        "id": item_id,
        "geometry": {"type": "Polygon", "coordinates": [[[lon, 0], [lon + 1, 0], [lon + 1, 1], [lon, 1], [lon, 0]]]},
        "properties": {"datetime": when},
    }


ROOT = "https://bucket/stac/catalog.json"
DOCS = {
    ROOT: {"links": [
        {"rel": "root", "href": "./catalog.json"},
        {"rel": "child", "href": "./2024/catalog.json", "title": "2024"},
        {"rel": "child", "href": "./2025/catalog.json", "title": "2025"},
    ]},
    "https://bucket/stac/2024/catalog.json": {"links": [
        {"rel": "child", "href": "./2024-12/catalog.json", "title": "2024-12"}]},
    "https://bucket/stac/2024/2024-12/catalog.json": {"links": [
        {"rel": "item", "href": "./a/a.json"}]},
    "https://bucket/stac/2025/catalog.json": {"links": [
        {"rel": "parent", "href": "../catalog.json"},
        {"rel": "child", "href": "./2025-01/catalog.json", "title": "2025-01"},
        {"rel": "child", "href": "./2025-03/catalog.json", "title": "2025-03"},
    ]},
    "https://bucket/stac/2025/2025-01/catalog.json": {"links": [
        {"rel": "item", "href": "./b/b.json"}, {"rel": "item", "href": "./skip/skip.json"},
        {"rel": "item", "href": "./c/c.json"}]},
    "https://bucket/stac/2025/2025-03/catalog.json": {"links": [{"rel": "item", "href": "./d/d.json"}]},
    "https://bucket/stac/2024/2024-12/a/a.json": _item("a", "2024-12-31T23:00:00Z", 0),
    "https://bucket/stac/2025/2025-01/b/b.json": _item("b", "2025-01-10T00:00:00Z", 0),
    "https://bucket/stac/2025/2025-01/c/c.json": _item("c", "2025-01-02T00:00:00Z", 50),
    "https://bucket/stac/2025/2025-03/d/d.json": _item("d", "2025-03-01T00:00:00Z", 0),
}


class _Response:
    def __init__(self, doc):
        self._doc = doc

    def raise_for_status(self):
        pass

    def json(self):
        return self._doc


class _Session:
    def __init__(self):
        self.urls = []

    def get(self, url, **kwargs):
        self.urls.append(url)
        return _Response(DOCS[url])


class _Connector(StaticSTACConnector):
    CATALOG = ROOT

    def __init__(self):
        self.session = _Session()

    def _get_session(self):
        return self.session

    def _wanted(self, href):
        return "skip" not in href

    def download(self, item, save_path):  # pragma: no cover
        raise NotImplementedError


def test_search_without_filters_reads_everything_oldest_first():
    assert [i["id"] for i in _Connector().search()] == ["a", "c", "b", "d"]


def test_search_prunes_date_catalogues_and_filters_items():
    connector = _Connector()
    items = connector.search(
        geojson={"type": "Polygon", "coordinates": [[[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8], [0.2, 0.2]]]},
        start_datetime=datetime.datetime(2025, 1, 1),  # naive: UTC
        stop_datetime=datetime.datetime(2025, 1, 31),
    )
    assert [i["id"] for i in items] == ["b"]  # c is elsewhere
    read = set(connector.session.urls)
    assert not any("2024" in u or "2025-03" in u for u in read)  # pruned unread
    assert not any("skip" in u for u in read)  # not wanted: never fetched


def test_search_by_ids():
    assert [i["id"] for i in _Connector().search(ids=["d", "a"])] == ["a", "d"]
    assert [i["id"] for i in _Connector().search(exclude_ids=["a", "b", "c"])] == ["d"]


def test_factory_knows_both_open_catalogues():
    assert get_connector("umbra-gec").CATALOG.endswith("/stac/catalog.json")
    assert "by-datetime" in get_connector("capella-gec").CATALOG
