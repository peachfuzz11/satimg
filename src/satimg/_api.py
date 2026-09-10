"""Top-level entry points, re-exported from :mod:`satimg`.

Kept out of ``satimg/__init__.py`` so that file is nothing but imports.
"""

from __future__ import annotations

import logging
import os
import tempfile
import zipfile
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version as _version
from typing import Iterator

from satimg.product import Product
from satimg.registry import resolve

# Library convention: log under the ``satimg.*`` namespace and let the
# application attach handlers. The NullHandler keeps us silent by default.
logging.getLogger("satimg").addHandler(logging.NullHandler())
logger = logging.getLogger("satimg")

try:
    __version__ = _version("satimg")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0"


def open(path: str) -> Product:
    """Open a product directory, returning the matching :class:`Product`."""
    cls = resolve(path)
    logger.debug("opened %s as %s", path, cls.__name__)
    return cls(path)


@contextmanager
def open_zip(zip_path: str, dest: str | None = None) -> Iterator[Product]:
    """Extract a zipped product to a temp dir and yield it as a :class:`Product`.

    Use the ``with`` form -- the temp dir (under ``dest``, or the system default)
    and the product's open rasters are both released on exit. Anything you need
    after the block must be pulled into memory first (``product.persist()``), or
    read while walking patches; a lazy view cannot be rebuilt once the temp dir
    is gone.
    """
    with tempfile.TemporaryDirectory(dir=dest, ignore_cleanup_errors=True) as tmp:
        logger.debug("extracting %s to %s", zip_path, tmp)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(tmp)
        entries = [os.path.join(tmp, e) for e in os.listdir(tmp)]
        if not entries:
            raise ValueError(f"{zip_path} extracted to nothing")
        root = entries[0] if len(entries) == 1 and os.path.isdir(entries[0]) else tmp
        logger.debug("extracted product root: %s", root)
        with open(root) as product:
            yield product
