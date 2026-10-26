"""Where a product's files live: a real directory, or an unextracted zip
archive.

Metadata files (XML / JSON manifests, thumbnails) are read through
:meth:`Source.open`; rasters are opened by ``rasterio``/``rioxarray`` (GDAL)
from :meth:`Source.raster_path` -- a plain file path for a directory, a
``/vsizip/`` path into the archive for a zip, which GDAL reads in place with
random access, nothing extracted. Every path a :class:`Source` method takes
is relative to the product's own root, ``"/"``-separated regardless of
platform.

:func:`open_source` is the factory :func:`~satimg.product.open` uses to pick
a :class:`DirSource` or a zip-native :class:`ZipSource` for a given path.
"""

from __future__ import annotations

import abc
import os
import zipfile
from typing import BinaryIO

from satimg.readers import find_file as _walk_find_file


class Source(abc.ABC):
    #: ``True`` for a real, already-extracted directory; ``False`` for an
    #: archive never written to disk.
    is_directory: bool

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The product's own identity string (its directory/root basename),
        for ``__repr__`` and :func:`~satimg.registry.resolve`."""

    @property
    @abc.abstractmethod
    def path(self) -> str:
        """The product directory, or just :attr:`name` when
        :attr:`is_directory` is ``False`` -- build raster paths with
        :meth:`raster_path`, never against this."""

    @abc.abstractmethod
    def find_file(self, filename: str) -> str | None:
        """Root-relative path of the first ``filename`` found anywhere under
        the root, or ``None``."""

    @abc.abstractmethod
    def listdir(self, subdir: str = "") -> list[str]:
        """Immediate children (files and directories) of ``subdir``, by
        name -- like :func:`os.listdir`."""

    @abc.abstractmethod
    def open(self, relpath: str) -> BinaryIO:
        """A binary, context-manager file object for ``relpath``."""

    @abc.abstractmethod
    def raster_path(self, relpath: str) -> str:
        """A path GDAL (``rasterio``/``rioxarray``) opens ``relpath`` by."""

    @abc.abstractmethod
    def exists(self, relpath: str) -> bool:
        """Whether ``relpath`` names a file under the root."""

    def close(self) -> None:
        """Release whatever this source itself opened (nothing, for a plain
        directory). Called by :meth:`~satimg.product.Product._close`;
        idempotent."""


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
        full = _walk_find_file(self._path, filename)
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


class ZipSource(Source):
    """An unextracted zip archive. Every read goes through the archive
    directly -- nothing is ever written to disk. Rasters are read by GDAL's
    ``/vsizip/``: a stored (uncompressed) member at plain file speed, a
    deflated one correctly but with each seek re-inflating from an earlier
    point -- write archives of already-compressed rasters with
    ``zipfile.ZIP_STORED``."""

    is_directory = False

    def __init__(self, archive: zipfile.ZipFile, root: str, display_name: str):
        """``root`` is the product's inner directory prefix (no leading/
        trailing ``/``), or ``""`` if the archive is flat (its own top level
        *is* the product root). ``display_name`` is what :attr:`name` returns.
        """
        self._archive = archive
        self._root = root.strip("/")
        self._display_name = display_name
        self._members = archive.namelist()

    @property
    def name(self) -> str:
        return self._display_name

    @property
    def path(self) -> str:
        return self._display_name

    def _full(self, relpath: str) -> str:
        relpath = relpath.strip("/")
        if not relpath:
            return self._root
        return f"{self._root}/{relpath}" if self._root else relpath

    def _rel(self, member: str) -> str | None:
        """``member``'s path relative to the root, or ``None`` if it's
        outside the root entirely."""
        if not self._root:
            return member
        if member in (self._root, self._root + "/"):
            return ""
        prefix = self._root + "/"
        return member[len(prefix):] if member.startswith(prefix) else None

    def find_file(self, filename: str) -> str | None:
        for member in self._members:
            rel = self._rel(member)
            if not rel or rel.endswith("/"):
                continue
            if rel.rsplit("/", 1)[-1] == filename:
                return rel
        return None

    def listdir(self, subdir: str = "") -> list[str]:
        subdir = subdir.strip("/")
        prefix = f"{subdir}/" if subdir else ""
        seen: list[str] = []
        seen_set = set()
        for member in self._members:
            rel = self._rel(member)
            if not rel:
                continue
            if prefix:
                if not rel.startswith(prefix):
                    continue
                tail = rel[len(prefix):]
            else:
                tail = rel
            if not tail:
                continue
            child = tail.split("/", 1)[0]
            if child not in seen_set:
                seen_set.add(child)
                seen.append(child)
        return seen

    def open(self, relpath: str) -> BinaryIO:
        return self._archive.open(self._full(relpath))

    def raster_path(self, relpath: str) -> str:
        # the braces delimit the archive's own path, so it needs no .zip
        # extension and may itself contain ".zip/"
        return f"/vsizip/{{{os.path.abspath(self._archive.filename)}}}/{self._full(relpath)}"

    def exists(self, relpath: str) -> bool:
        return self._full(relpath) in self._members

    def close(self) -> None:
        self._archive.close()


def zip_source(zip_path: str) -> ZipSource:
    """A zip-native :class:`ZipSource` for the archive at ``zip_path``.

    Opens the ``zipfile.ZipFile`` itself (not handed one already open) and
    detects the product's inner root the same way :func:`open_zip` does: the
    single top-level directory the archive extracts to, or -- for a flat
    archive, multiple top-level entries -- the archive's own top level.
    Released by the returned :class:`ZipSource`'s :meth:`~Source.close`.
    """
    archive = zipfile.ZipFile(zip_path)
    try:
        names = archive.namelist()
        top = {n.split("/", 1)[0] for n in names if n.strip("/")}
        if not top:
            raise ValueError(f"{zip_path} contains nothing")
        only = next(iter(top)) if len(top) == 1 else None
        root = only if only is not None and any(
            n.startswith(only + "/") for n in names
        ) else ""
        display_name = root or os.path.splitext(os.path.basename(zip_path))[0]
    except Exception:
        archive.close()
        raise
    return ZipSource(archive, root, display_name)


def open_source(path: str) -> Source:
    """The :class:`Source` for ``path`` -- a :class:`DirSource` for a
    directory, or a zip-native :class:`ZipSource` (see :func:`zip_source`,
    nothing extracted) for a zip archive. A zip is detected by content
    (:func:`zipfile.is_zipfile`), not by extension."""
    if zipfile.is_zipfile(path):
        return zip_source(path)
    return DirSource(path)
