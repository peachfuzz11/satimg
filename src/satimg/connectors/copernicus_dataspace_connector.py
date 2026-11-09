"""The Copernicus Data Space connector the Sentinel-1, Sentinel-2 and
Sentinel-3 connectors share.

Downloads need a (free) Copernicus Data Space account. Pass it explicitly::

    Sentinel2L1CConnector(username="me@example.com", password="...")

or set ``CDSE_USERNAME`` / ``CDSE_PASSWORD`` in the environment.
"""

from __future__ import annotations

import logging
import os
import zipfile

import requests

from satimg.connectors._credentials import Credentials
from satimg.connectors.connector import STACConnector

logger = logging.getLogger(__name__)

_REDIRECTS = (301, 302, 303, 307, 308)

_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
)


class CopernicusDataspaceConnector(STACConnector):
    STAC_ENDPOINT = "https://stac.dataspace.copernicus.eu/v1/"
    ENV_PREFIX = "CDSE"
    SERVICE = "Copernicus Data Space"

    def __init__(self, username: str | None = None, password: str | None = None):
        self._username = username
        self._password = password

    def _credentials(self) -> Credentials:
        return Credentials.resolve(
            self._username,
            self._password,
            service=self.SERVICE,
            env_prefix=self.ENV_PREFIX,
        )

    def download(self, item: dict, save_path: str) -> str:
        """Stream the product's zip to ``save_path`` (a Sentinel-1 GRD is a
        GB or two, so never held in memory), via a ``.part`` file so a failed
        or corrupt download never leaves a truncated product behind."""
        logger.info("downloading %s -> %s", item["id"], save_path)
        session = self._get_session()
        # "Product" for Sentinel-1 / -2 items, "product" for Sentinel-3
        assets = item["assets"]
        url = (assets.get("Product") or assets["product"])["href"]
        # follow redirects by hand: requests drops the bearer token on a
        # redirect to another host, and the download host needs it
        response = session.get(url, allow_redirects=False, stream=True, timeout=300)
        while response.status_code in _REDIRECTS:
            response.close()
            url = response.headers["Location"]
            logger.debug("redirect %d -> %s", response.status_code, url)
            response = session.get(url, allow_redirects=False, stream=True, timeout=300)

        partial = save_path + ".part"
        try:
            with response:
                response.raise_for_status()
                with open(partial, "wb") as f:
                    for block in response.iter_content(chunk_size=8 << 20):
                        f.write(block)
            with zipfile.ZipFile(partial) as archive:
                if archive.testzip() is not None:
                    raise zipfile.BadZipFile("invalid zip archive")
        except BaseException:
            if os.path.exists(partial):
                os.remove(partial)
            raise
        os.replace(partial, save_path)
        logger.info("wrote %s (%d bytes)", save_path, os.path.getsize(save_path))
        return save_path

    def _get_session(self, *args, **kwargs) -> requests.Session:
        creds = self._credentials()
        logger.debug("requesting %s access token for %s", self.SERVICE, creds.username)
        response = requests.post(
            _TOKEN_URL,
            data={
                "client_id": "cdse-public",
                "username": creds.username,
                "password": creds.password,
                "grant_type": "password",
            },
        )
        response.raise_for_status()
        session = requests.Session()
        session.headers["Authorization"] = f"Bearer {response.json()['access_token']}"
        logger.debug("obtained %s access token", self.SERVICE)
        return session
