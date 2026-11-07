"""Products on the local filesystem: a directory, or a single file."""

from __future__ import annotations

import os
from typing import BinaryIO

from satimg.source.base import Source


def _walk(directory: str, filename: str) -> str | None:
    """The full path of the first ``filename`` found under ``directory``."""
    for root, _dirs, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None


class DirSource(Source):
    """A product directory on disk -- byte-identical semantics to the raw
    ``os.*`` calls this replaces."""

    is_directory = True

    def __init__(self, path: str):
        self._path = str(path)

    @property
    def name(self) -> str:
        return os.path.basename(os.path.normpath(self._path))

    @property
    def path(self) -> str:
        return self._path

    def _abs(self, relpath: str) -> str:
        return os.path.join(self._path, *relpath.split("/")) if relpath else self._path

    def find_file(self, filename: str) -> str | None:
        full = _walk(self._path, filename)
        if full is None:
            return None
        return os.path.relpath(full, self._path).replace(os.sep, "/")

    def listdir(self, subdir: str = "") -> list[str]:
        return os.listdir(self._abs(subdir))

    def open(self, relpath: str) -> BinaryIO:
        return open(self._abs(relpath), "rb")

    def raster_path(self, relpath: str) -> str:
        return self._abs(relpath)

    def exists(self, relpath: str) -> bool:
        return os.path.isfile(self._abs(relpath))


class FileSource(DirSource):
    """A product that is a single file (e.g. a NISAR GCOV or VIIRS L1B
    ``.h5`` / ``.nc``) rather than a directory. Its root is the file's own directory, so
    the file and any sibling it pairs with (a VIIRS geolocation file) are
    reached by their bare names; :attr:`name` and :attr:`path` are the file's
    own, so :func:`~satimg.registry.resolve` matches the file name.

    Unlike a :class:`DirSource`, :meth:`find_file` looks only at the file's
    immediate siblings -- the directory a single file sits in (a downloads
    folder, say) is not the product, so walking it would be wasted work.
    """

    def __init__(self, path: str):
        self._file = os.path.abspath(str(path))
        super().__init__(os.path.dirname(self._file))

    @property
    def name(self) -> str:
        return os.path.basename(self._file)

    @property
    def path(self) -> str:
        return self._file

    def find_file(self, filename: str) -> str | None:
        return filename if self.exists(filename) else None
