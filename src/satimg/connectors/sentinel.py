"""Copernicus Data Space connectors for Sentinel-1 and Sentinel-2.

Downloads need a (free) Copernicus Data Space account. Pass it explicitly::

    Sentinel2Connector(username="me@example.com", password="...")

or set ``CDSE_USERNAME`` / ``CDSE_PASSWORD`` in the environment.
"""

from __future__ import annotations

import io
import zipfile

import requests

from satimg.connectors._credentials import Credentials
from satimg.connectors.base import STACConnector

_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
)


class _DataspaceConnector(STACConnector):
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
        session = self._get_session()
        url = item["assets"]["Product"]["href"]
        response = session.get(url, allow_redirects=False)
        while response.status_code in (301, 302, 303, 307):
            url = response.headers["Location"]
            response = session.get(url, allow_redirects=False)

        payload = session.get(url, allow_redirects=True).content
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if archive.testzip() is not None:
                raise zipfile.BadZipFile("invalid zip archive")
        with open(save_path, "wb") as f:
            f.write(payload)
        return save_path

    def _get_session(self, *args, **kwargs) -> requests.Session:
        creds = self._credentials()
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
        return session


class Sentinel1Connector(_DataspaceConnector):
    collection = "sentinel-1-grd"
    product_type = "IW_GRDH_1S"


class Sentinel2Connector(_DataspaceConnector):
    collection = "sentinel-2-l1c"
