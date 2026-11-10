"""Connector base classes: search a STAC catalogue, download a product."""

from __future__ import annotations

import abc
import datetime
import logging
import os
import zipfile

import requests

logger = logging.getLogger(__name__)


def stream(session: requests.Session, url: str, out) -> None:
    """Copy ``url`` into the binary file object ``out``, block by block."""
    with session.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        for block in response.iter_content(chunk_size=8 << 20):
            out.write(block)


def stream_to_zip(session: requests.Session, members: dict[str, str], save_path: str) -> str:
    """Stream each ``{archive name: url}`` of ``members`` into a zip at
    ``save_path``, members *stored* -- rasters are compressed already, and
    GDAL reads stored members in place at file speed (see
    :class:`~satimg.source.zip_source.ZipSource`). Written as a ``.part``
    file and only renamed into place once whole, so a failed download never
    leaves a truncated product behind."""
    partial = save_path + ".part"
    try:
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_STORED) as archive:
            for name, url in members.items():
                logger.debug("fetching %s: %s", name, url)
                with archive.open(name, "w", force_zip64=True) as out:
                    stream(session, url, out)
    except BaseException:
        if os.path.exists(partial):
            os.remove(partial)
        raise
    os.replace(partial, save_path)
    logger.info("wrote %s", save_path)
    return save_path


class Connector(abc.ABC):
    """Find and fetch products from a remote archive."""

    @abc.abstractmethod
    def search(
        self,
        geojson: dict | None = None,
        start_datetime: datetime.datetime | None = None,
        stop_datetime: datetime.datetime | None = None,
        **kwargs,
    ) -> list[dict]:
        ...

    @abc.abstractmethod
    def download(self, item: dict, save_path: str) -> str:
        ...


class STACConnector(Connector, abc.ABC):
    """A :class:`Connector` backed by a pystac-client ``Client``.

    ``search`` is anonymous; subclasses add credentials only for ``download``.
    The STAC client is opened lazily on first ``search`` so a connector can be
    constructed offline.
    """

    STAC_ENDPOINT: str | None = None
    collection: str | None = None
    product_type: str | None = None

    @property
    def client(self):
        client = getattr(self, "_client", None)
        if client is None:
            from pystac_client import Client

            logger.debug("opening STAC client: %s", self.STAC_ENDPOINT)
            client = self._client = Client.open(self.STAC_ENDPOINT)
        return client

    def search(
        self,
        geojson: dict | None = None,
        start_datetime: datetime.datetime | None = None,
        stop_datetime: datetime.datetime | None = None,
        ids: list[str] | None = None,
        exclude_ids: list[str] | None = None,
        **kwargs,
    ) -> list[dict]:
        if start_datetime and stop_datetime:
            dt = f"{start_datetime.isoformat()}/{stop_datetime.isoformat()}"
        elif start_datetime:
            dt = f"{start_datetime.isoformat()}/"
        elif stop_datetime:
            dt = f"/{stop_datetime.isoformat()}"
        else:
            dt = None

        query_filter = None
        if self.product_type and not ids:
            query_filter = {
                "filter": {
                    "op": "and",
                    "args": [
                        {"op": "=", "args": [{"property": "product:type"}, self.product_type]}
                    ],
                }
            }

        logger.debug(
            "STAC search: collection=%s datetime=%s ids=%s", self.collection, dt, ids
        )
        items = [
            item.to_dict()
            for item in self.client.search(
                collections=[self.collection],
                datetime=dt,
                intersects=geojson,
                ids=ids,
                filter=query_filter,
            ).items()
        ]
        if exclude_ids:
            kept = [i for i in items if i["id"] not in exclude_ids]
            logger.debug(
                "STAC search returned %d item(s), %d after exclude_ids",
                len(items),
                len(kept),
            )
            return kept
        logger.debug("STAC search returned %d item(s)", len(items))
        return items

    @abc.abstractmethod
    def _get_session(self, *args, **kwargs):
        ...
