"""The :class:`Source` interface: where a product's files live.

Metadata files (XML / JSON manifests, thumbnails) are read through
:meth:`Source.open`; rasters are opened by ``rasterio``/``rioxarray`` (GDAL)
from :meth:`Source.raster_path`. Every path a :class:`Source` method takes is
relative to the product's own root, ``"/"``-separated regardless of
platform.
"""

from __future__ import annotations

import abc
from typing import BinaryIO


class Source(abc.ABC):
    """Read access to one product's files, wherever they are stored."""

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

    def open_detached(self, relpath: str) -> BinaryIO:
        """Like :meth:`open`, but a handle of its own that does not depend on
        this source staying open -- for lazy readers that reopen a file after
        the product (and its source) were closed, as GDAL does by path."""
        return self.open(relpath)

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
