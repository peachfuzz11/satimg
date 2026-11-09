"""NISAR connector: search the ASF DAAC, download from NASA Earthdata.

``search`` is anonymous (the ASF search API). ``download`` needs a free NASA
Earthdata Login: an EDL bearer token, or a username and password. Pass them
explicitly::

    NisarGCOVConnector(token="eyJ...")
    NisarGCOVConnector(username="me", password="...")

or set ``EARTHDATA_TOKEN`` -- or ``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD``
-- in the environment.

Unlike the Copernicus / USGS connectors, ``download`` writes the product's
own single ``.h5`` file, not a zip: ``satimg.open`` reads it as it is.
"""

from __future__ import annotations

import datetime
import logging
import os

import requests

from satimg.connectors.connector import Connector, stream
from satimg.connectors.nasa_earthdata_connector import EarthdataLogin
from satimg.geojson import geometry_of

logger = logging.getLogger(__name__)

SEARCH_URL = "https://api.daac.asf.alaska.edu/services/search/param"


def _wkt(geojson: dict) -> str:
    """A GeoJSON Polygon / Feature / single-feature FeatureCollection as WKT."""
    geojson = geometry_of(geojson)
    if geojson["type"] == "Point":
        lon, lat = geojson["coordinates"][:2]
        return f"POINT({lon} {lat})"
    if geojson["type"] != "Polygon":
        raise ValueError(f"unsupported geometry for NISAR search: {geojson['type']}")
    ring = ", ".join(f"{lon} {lat}" for lon, lat, *_ in geojson["coordinates"][0])
    return f"POLYGON(({ring}))"


class NisarGCOVConnector(EarthdataLogin, Connector):
    """NISAR L-band products from the ASF DAAC; GCOV by default (pass
    ``processing_level=`` for another, e.g. ``"RSLC"`` -- though only GCOV
    opens as a product)."""

    collection = "nisar-gcov"

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        token: str | None = None,
        processing_level: str = "GCOV",
    ):
        super().__init__(username, password, token)
        self.processing_level = processing_level

    def search(
        self,
        geojson: dict | None = None,
        start_datetime: datetime.datetime | None = None,
        stop_datetime: datetime.datetime | None = None,
        ids: list[str] | None = None,
        exclude_ids: list[str] | None = None,
        max_results: int = 250,
        **kwargs,
    ) -> list[dict]:
        """GeoJSON Features, one per granule, STAC-shaped enough to share
        code with the other connectors: ``item["id"]`` is the granule name and
        ``item["assets"]["data"]["href"]`` its ``.h5`` URL. Extra keyword
        arguments go to the ASF search API as-is (e.g.
        ``flightDirection="ASCENDING"``, ``relativeOrbit=5``)."""
        params = {
            "dataset": "NISAR",
            "processingLevel": self.processing_level,
            "output": "geojson",
            "maxResults": max_results,
            **kwargs,
        }
        if geojson is not None:
            params["intersectsWith"] = _wkt(geojson)
        if start_datetime:
            params["start"] = start_datetime.isoformat()
        if stop_datetime:
            params["end"] = stop_datetime.isoformat()
        if ids:
            params["granule_list"] = ",".join(ids)
        logger.debug("ASF search: %s", params)
        response = requests.get(SEARCH_URL, params=params, timeout=120)
        response.raise_for_status()
        items = []
        for feature in response.json()["features"]:
            props = feature["properties"]
            items.append({
                **feature,
                "id": props["sceneName"],
                "assets": {"data": {"href": props["url"]}},
            })
        if exclude_ids:
            items = [i for i in items if i["id"] not in exclude_ids]
        logger.debug("ASF search returned %d item(s)", len(items))
        return items

    def download(self, item: dict, save_path: str) -> str:
        """Stream the granule's ``.h5`` to ``save_path`` (a GCOV frame is
        several GB), via a ``.part`` file so an interrupted download never
        leaves a truncated product behind."""
        url = item["assets"]["data"]["href"]
        logger.info("downloading %s -> %s", item["id"], save_path)
        session = self._get_session()
        partial = save_path + ".part"
        with open(partial, "wb") as f:
            stream(session, url, f)
        os.replace(partial, save_path)
        logger.info("wrote %s", save_path)
        return save_path
