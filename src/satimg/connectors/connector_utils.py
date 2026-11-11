"""Download helpers the connectors share: streaming a URL to a file, or
several into a zip."""

from __future__ import annotations

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
