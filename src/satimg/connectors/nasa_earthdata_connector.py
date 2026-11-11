"""NASA Earthdata: Earthdata Login (EDL) sessions, and a connector base that
searches NASA's Common Metadata Repository (CMR).

CMR search is anonymous. Downloads from a NASA DAAC (ASF, LAADS, ...) need a
free Earthdata Login: an EDL bearer token, or a username and password --
passed to the connector (``token=`` / ``username=`` / ``password=``), or from
``EARTHDATA_TOKEN`` / ``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD``.
"""

from __future__ import annotations

import datetime
import logging
import os
import zipfile
from urllib.parse import urlparse

import requests

from satimg.connectors._credentials import Credentials, CredentialsError
from satimg.connectors.connector import Connector
from satimg.connectors.connector_utils import stream
from satimg.geojson import bbox

logger = logging.getLogger(__name__)

EDL_HOST = "urs.earthdata.nasa.gov"
CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.json"


class EarthdataSession(requests.Session):
    """Keeps basic auth on the redirect to Earthdata Login (and only there):
    ``requests`` drops credentials on any cross-host redirect, which is
    exactly the hop EDL's OAuth flow needs them on; and a bearer token must
    not follow the DAAC's redirect on to signed cloud storage."""

    def rebuild_auth(self, prepared_request, response):
        if "Authorization" in prepared_request.headers:
            new = urlparse(prepared_request.url).hostname
            old = urlparse(response.request.url).hostname
            if new != old and EDL_HOST not in (new, old):
                del prepared_request.headers["Authorization"]


class EarthdataLogin:
    """Mixin: Earthdata Login credentials -> an authenticated session."""

    ENV_PREFIX = "EARTHDATA"
    SERVICE = "NASA Earthdata"

    def __init__(self, username: str | None = None, password: str | None = None, token: str | None = None):
        self._username = username
        self._password = password
        self._token = token

    def _get_session(self, *args, **kwargs) -> requests.Session:
        session = EarthdataSession()
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


def _bbox(geojson: dict) -> str:
    """CMR's ``bounding_box`` (``W,S,E,N``) around a GeoJSON geometry /
    Feature / single-feature FeatureCollection."""
    return ",".join(str(v) for v in bbox(geojson))


def _file_url(entry: dict, suffix: str) -> str | None:
    """A CMR granule entry's HTTPS download link ending in ``suffix``."""
    for link in entry.get("links", []):
        href = link.get("href", "")
        if href.startswith("https://") and href.endswith(suffix) and "opendap" not in href:
            return href
    return None


class CMRConnector(EarthdataLogin, Connector):
    """Granules of NASA collections, found through CMR (anonymous) and
    downloaded from their DAAC (Earthdata Login).

    A subclass sets :attr:`SATELLITES` -- each satellite's short names,
    several when a product is split over files (data + geolocation) that
    pair by :meth:`_pair_key` -- and :attr:`suffix`, the file type to fetch.
    ``satellite="both"`` (the default) searches every satellite, else just
    the one named. ``search`` yields one item per granule with an asset per
    short name; ``download`` writes them all into one stored zip under a
    folder named for the item.
    """

    #: satellite -> the short names whose files pair into one granule
    SATELLITES: dict[str, tuple[str, ...]] = {}
    suffix = ".nc"
    #: CMR ``version`` to restrict to, if any
    version: str | None = None

    def __init__(
        self,
        satellite: str = "both",
        username: str | None = None,
        password: str | None = None,
        token: str | None = None,
    ):
        if satellite == "both":
            self._satellites = tuple(self.SATELLITES)
        elif satellite in self.SATELLITES:
            self._satellites = (satellite,)
        else:
            raise ValueError(f"unknown satellite {satellite!r}; known: both, {', '.join(self.SATELLITES)}")
        super().__init__(username, password, token)
        self.satellite = satellite

    @property
    def short_names(self) -> tuple[str, ...]:
        """Every short name searched, across :meth:`_groups`."""
        return tuple(name for group in self._groups() for name in group)

    def _pair_key(self, filename: str) -> str:
        """What a granule's files share across :attr:`short_names`; by
        default everything from the first ``.`` on (``.A2026305.1630.021``)
        up to the production time."""
        parts = filename.split(".")
        return ".".join(parts[1:4])

    def _groups(self) -> list[tuple[str, ...]]:
        """The products to search: each a tuple of short names whose files
        pair into one granule -- one per chosen satellite (NOAA-20 / NOAA-21,
        Terra / Aqua)."""
        return [self.SATELLITES[s] for s in self._satellites]

    def _search_group(self, short_names: tuple[str, ...], params: dict) -> list[dict]:
        found: dict[str, dict] = {}
        for short_name in short_names:
            for entry in self._query(short_name, params):
                url = _file_url(entry, self.suffix)
                if url is None:
                    continue
                name = url.rsplit("/", 1)[-1]
                item = found.setdefault(self._pair_key(name), {"assets": {}, "entry": entry})
                item["assets"][short_name] = {"href": url}
        items = []
        for item in found.values():
            if len(item["assets"]) < len(short_names):
                continue  # a granule missing one of its files
            entry = item.pop("entry")
            first = item["assets"][short_names[0]]["href"].rsplit("/", 1)[-1]
            item.update({
                "id": first[: -len(self.suffix)],
                "type": "Feature",
                "properties": {
                    "datetime": entry.get("time_start"),
                    "day_night": (entry.get("day_night_flag") or "").lower() or None,
                },
                "geometry": None,
            })
            items.append(item)
        return items

    def _query(self, short_name: str, params: dict) -> list[dict]:
        entries, page = [], 1
        while True:
            query = {**params, "short_name": short_name, "page_size": 2000, "page_num": page}
            if self.version:
                query["version"] = self.version
            response = requests.get(CMR_GRANULES, params=query, timeout=120)
            response.raise_for_status()
            batch = response.json()["feed"]["entry"]
            entries += batch
            if len(batch) < 2000:
                return entries
            page += 1

    def search(
        self,
        geojson: dict | None = None,
        start_datetime: datetime.datetime | None = None,
        stop_datetime: datetime.datetime | None = None,
        ids: list[str] | None = None,
        exclude_ids: list[str] | None = None,
        day_night: str | None = None,
        **kwargs,
    ) -> list[dict]:
        """Granules overlapping ``geojson``'s bounding box and the time
        window; ``day_night`` (``"day"`` / ``"night"`` / ``"both"``) filters
        on CMR's day/night flag. Extra keyword arguments go to CMR as-is.
        ``item["id"]`` is the first short name's file stem."""
        params = dict(kwargs)
        if geojson is not None:
            params["bounding_box"] = _bbox(geojson)
        if start_datetime or stop_datetime:
            fmt = "%Y-%m-%dT%H:%M:%SZ"
            params["temporal"] = (
                f"{start_datetime.strftime(fmt) if start_datetime else ''},"
                f"{stop_datetime.strftime(fmt) if stop_datetime else ''}"
            )
        if day_night:
            params["day_night_flag"] = day_night
        items = []
        for group in self._groups():
            items += self._search_group(group, params)
        if ids:
            items = [i for i in items if i["id"] in ids]
        if exclude_ids:
            items = [i for i in items if i["id"] not in exclude_ids]
        items.sort(key=lambda i: (i["properties"]["datetime"] or "", i["id"]))
        logger.debug("CMR search returned %d granule(s)", len(items))
        return items

    def download(self, item: dict, save_path: str) -> str:
        """Every asset into one stored zip, under a folder named for the
        item -- which ``satimg.open`` reads zip-native."""
        logger.info("downloading %s -> %s", item["id"], save_path)
        session = self._get_session()
        partial = save_path + ".part"
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_STORED) as archive:
            for asset in item["assets"].values():
                name = asset["href"].rsplit("/", 1)[-1]
                with archive.open(f"{item['id']}/{name}", "w", force_zip64=True) as out:
                    stream(session, asset["href"], out)
        os.replace(partial, save_path)
        logger.info("wrote %s", save_path)
        return save_path
