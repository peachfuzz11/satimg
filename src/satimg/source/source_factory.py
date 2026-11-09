"""Picking the :class:`~satimg.source.source.Source` for a path."""

from __future__ import annotations

import os
import zipfile

from satimg.source.zip_source import zip_source
from satimg.source.source import Source
from satimg.source.local_source import DirSource, FileSource


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
