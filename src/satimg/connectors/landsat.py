"""USGS LandsatLook connector.

Downloads need a USGS EarthExplorer / ERS account. Pass it explicitly::

    LandsatConnector(username="me", password="...", token="...")

or set ``USGS_USERNAME`` / ``USGS_PASSWORD`` (and optionally ``USGS_TOKEN``) in
the environment.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

from satimg.connectors._auth import ers_login
from satimg.connectors._credentials import Credentials
from satimg.connectors.base import STACConnector

_ASSETS = (
    "thumbnail", "MTL.json", "coastal", "blue", "green", "red", "nir08", "swir16",
    "swir22", "pan", "cirrus", "lwir11", "lwir12", "qa_pixel", "qa_radsat",
    "ANG.txt", "VAA", "VZA", "SAA", "SZA",
)


class LandsatConnector(STACConnector):
    STAC_ENDPOINT = "https://landsatlook.usgs.gov/stac-server"
    collection = "landsat-c2l1"
    ENV_PREFIX = "USGS"
    SERVICE = "USGS EarthExplorer"

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        token: str | None = None,
    ):
        self._username = username
        self._password = password
        self._token = token

    def _get_session(self, *args, **kwargs):
        creds = Credentials.resolve(
            self._username,
            self._password,
            self._token,
            service=self.SERVICE,
            env_prefix=self.ENV_PREFIX,
        )
        return ers_login(creds.username, creds.password, creds.token)

    def download(self, item: dict, save_path: str) -> str:
        session = self._get_session()
        with zipfile.ZipFile(save_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, asset in item["assets"].items():
                if name not in _ASSETS:
                    continue
                href = asset["href"]
                filename = name if Path(name).suffix else name + Path(href).suffix
                response = session.get(href, timeout=120)
                response.raise_for_status()
                archive.writestr(os.path.join(item["id"], filename), response.content)
            if archive.testzip() is not None:
                raise zipfile.BadZipFile("invalid zip archive")
        return save_path
