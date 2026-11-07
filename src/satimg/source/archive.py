"""Products read in place from an unextracted zip archive."""

from __future__ import annotations

import os
import zipfile
from typing import BinaryIO

from satimg.source.base import Source


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

    def open_detached(self, relpath: str) -> BinaryIO:
        archive = zipfile.ZipFile(self._archive.filename)
        member = archive.open(self._full(relpath))
        close_member = member.close

        def close():
            close_member()
            archive.close()

        member.close = close
        return member

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
    detects the product's inner root the same way :func:`~satimg.product.open_zip` does: the
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
