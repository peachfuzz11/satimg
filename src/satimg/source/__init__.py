"""Where a product's files live.

- :class:`DirSource` -- an extracted product directory
- :class:`FileSource` -- a product that is one file (a NISAR or VIIRS HDF5),
  reaching the siblings it pairs with
- :class:`ZipSource` -- an unextracted zip, read in place (rasters through
  GDAL's ``/vsizip/``), nothing written to disk

:func:`open_source` is the factory :func:`~satimg.product.open` uses to pick
one for a path.
"""

from __future__ import annotations

import os
import zipfile

from satimg.source.archive import ZipSource, zip_source
from satimg.source.base import Source
from satimg.source.local import DirSource, FileSource

__all__ = ["Source", "DirSource", "FileSource", "ZipSource", "open_source", "zip_source"]


def open_source(path: str) -> Source:
    """The :class:`Source` for ``path`` -- a :class:`DirSource` for a
    directory, a zip-native :class:`ZipSource` (see :func:`zip_source`,
    nothing extracted) for a zip archive, or a :class:`FileSource` for any
    other single file. A zip is detected by content
    (:func:`zipfile.is_zipfile`), not by extension."""
    if zipfile.is_zipfile(path):
        return zip_source(path)
    if os.path.isfile(path):
        return FileSource(path)
    return DirSource(path)
