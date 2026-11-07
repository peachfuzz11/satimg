"""NISAR connector: search the ASF DAAC, download from NASA Earthdata.

``search`` is anonymous (the ASF search API). ``download`` needs a free NASA
Earthdata Login: an EDL bearer token, or a username and password. Pass them
explicitly::

    NisarConnector(token="eyJ...")
    NisarConnector(username="me", password="...")

or set ``EARTHDATA_TOKEN`` -- or ``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD``
-- in the environment.

Unlike the Copernicus / USGS connectors, ``download`` writes the product's
own single ``.h5`` file, not a zip: ``satimg.open`` reads it as it is.
"""

from __future__ import annotations

import datetime
import logging
import os
from urllib.parse import urlparse

import requests

from satimg.connectors._credentials import Credentials, CredentialsError
from satimg.connectors.base import Connector

logger = logging.getLogger(__name__)

SEARCH_URL = "https://api.daac.asf.alaska.edu/services/search/param"
EDL_HOST = "urs.earthdata.nasa.gov"


class _EarthdataSession(requests.Session):
    """Keeps basic auth on the redirect to Earthdata Login (and only there):
    ``requests`` drops credentials on any cross-host redirect, which is
    exactly the hop EDL's OAuth flow needs them on."""

    def rebuild_auth(self, prepared_request, response):
        if "Authorization" in prepared_request.headers:
            new = urlparse(prepared_request.url).hostname
            old = urlparse(response.request.url).hostname
            if new != old and EDL_HOST not in (new, old):
                del prepared_request.headers["Authorization"]


def _wkt(geojson: dict) -> str:
    """A GeoJSON Polygon / Feature / single-feature FeatureCollection as WKT."""
    if geojson.get("type") == "FeatureCollection":
        geojson = geojson["features"][0]
    if geojson.get("type") == "Feature":
        geojson = geojson["geometry"]
    if geojson["type"] == "Point":
        lon, lat = geojson["coordinates"][:2]
        return f"POINT({lon} {lat})"
    if geojson["type"] != "Polygon":
        raise ValueError(f"unsupported geometry for NISAR search: {geojson['type']}")
    ring = ", ".join(f"{lon} {lat}" for lon, lat, *_ in geojson["coordinates"][0])
    return f"POLYGON(({ring}))"


class NisarConnector(Connector):
    """NISAR L-band products from the ASF DAAC; GCOV by default (pass
    ``processing_level=`` for another, e.g. ``"RSLC"`` -- though only GCOV
    opens as a product)."""

    ENV_PREFIX = "EARTHDATA"
    SERVICE = "NASA Earthdata"
    collection = "nisar-gcov"

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        token: str | None = None,
        processing_level: str = "GCOV",
    ):
        self._username = username
        self._password = password
        self._token = token
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

    def _get_session(self, *args, **kwargs) -> requests.Session:
        session = _EarthdataSession()
        token = self._token or os.environ.get(f"{self.ENV_PREFIX}_TOKEN")
        if token:
            logger.debug("using an %s bearer token", self.SERVICE)
            session.headers["Authorization"] = f"Bearer {token}"
            return session
        try:
            creds = Credentials.resolve(
                self._username, self._password, service=self.SERVICE, env_prefix=self.ENV_PREFIX
            )
        except CredentialsError as e:
            raise CredentialsError(
                f"{e} Or pass token= / set {self.ENV_PREFIX}_TOKEN to an Earthdata Login bearer token."
            ) from None
        session.auth = (creds.username, creds.password)
        return session

    def download(self, item: dict, save_path: str) -> str:
        """Stream the granule's ``.h5`` to ``save_path`` (a GCOV frame is
        several GB), via a ``.part`` file so an interrupted download never
        leaves a truncated product behind."""
        url = item["assets"]["data"]["href"]
        logger.info("downloading %s -> %s", item["id"], save_path)
        session = self._get_session()
        partial = save_path + ".part"
        with session.get(url, stream=True, timeout=300) as response:
            response.raise_for_status()
            with open(partial, "wb") as f:
                for block in response.iter_content(chunk_size=8 << 20):
                    f.write(block)
        os.replace(partial, save_path)
        logger.info("wrote %s", save_path)
        return save_path
