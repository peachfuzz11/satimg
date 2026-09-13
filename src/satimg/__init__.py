"""satimg -- a uniform, tile-friendly wrapper over satellite imagery.

    import satimg

    product = satimg.open("/data/S2A_MSIL1C_....SAFE")

    for patch in product.patches(512, overlap=64):
        do_something(patch.raw.values)    # (band, y, x) native dtype
        print(patch.col, patch.row)       # where it sits in the full image
        print(patch.meta.sample())        # {field: value} at the patch centre
"""

import logging
from importlib.metadata import PackageNotFoundError, version as _version

# Log under the ``satimg.*`` namespace; the app attaches handlers.
logging.getLogger(__name__).addHandler(logging.NullHandler())

try:
    __version__ = _version("satimg")
except PackageNotFoundError:  # a source tree without an install
    __version__ = "0.0.0"

from satimg import products as _products  # noqa: F401  (populates the registry)
from satimg.connectors import CredentialsError, get_connector
from satimg.geometry import Grid, Window, windows_at
from satimg.metadata import Field, Metadata
from satimg.patch import Patch
from satimg.product import Product, open, open_zip
from satimg.registry import UnknownProductError
from satimg.tiling import as_band_yx, label_bands, read_window

__all__ = [
    "__version__",
    "open",
    "open_zip",
    "get_connector",
    "Product",
    "Patch",
    "read_window",
    "as_band_yx",
    "label_bands",
    "Window",
    "Grid",
    "windows_at",
    "Metadata",
    "Field",
    "UnknownProductError",
    "CredentialsError",
]
