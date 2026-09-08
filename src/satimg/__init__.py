"""satimg -- a uniform, tile-friendly wrapper over satellite imagery.

    import satimg

    product = satimg.open("/data/S2A_MSIL1C_....SAFE")

    for patch in product.patches(512, overlap=64):
        do_something(patch.values)        # (band, y, x) native dtype
        print(patch.col, patch.row)       # where it sits in the full image
        print(patch.center_latlon)        # where it sits on Earth

    # any other array -- the uint8 visualisation, a derived index, ...
    for patch in satimg.patches(product.visual, 512, transformer=product.transformer):
        ...
"""

from __future__ import annotations

import logging
import os
import tempfile
import zipfile
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version as _version
from typing import Iterator

# Library convention: log under the ``satimg.*`` namespace and let the
# application attach handlers. The NullHandler keeps us silent by default.
logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

try:
    __version__ = _version("satimg")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0"

from satimg import products as _products  # noqa: F401  (populates the registry)
from satimg.connectors import CredentialsError, get_connector
from satimg.geometry import Grid, Window, windows_at
from satimg.metadata import Field, Metadata
from satimg.product import Product
from satimg.registry import UnknownProductError, resolve
from satimg.tiling import Patch, patches, patches_at, read_window

__all__ = [
    "__version__",
    "open",
    "open_zip",
    "get_connector",
    "Product",
    "Patch",
    "patches",
    "patches_at",
    "read_window",
    "Window",
    "Grid",
    "windows_at",
    "Metadata",
    "Field",
    "UnknownProductError",
    "CredentialsError",
]


def open(path: str) -> Product:
    """Open a product directory, returning the matching :class:`Product`."""
    cls = resolve(path)
    logger.debug("opened %s as %s", path, cls.__name__)
    return cls(path)


@contextmanager
def open_zip(zip_path: str, dest: str | None = None) -> Iterator[Product]:
    """Extract a zipped product to a temp dir and yield it as a :class:`Product`.

    The temp dir (under ``dest``, or the system default) is removed on exit.
    """
    with tempfile.TemporaryDirectory(dir=dest) as tmp:
        logger.debug("extracting %s to %s", zip_path, tmp)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(tmp)
        entries = [os.path.join(tmp, e) for e in os.listdir(tmp)]
        root = entries[0] if len(entries) == 1 and os.path.isdir(entries[0]) else tmp
        logger.debug("extracted product root: %s", root)
        yield open(root)
