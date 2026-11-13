"""Map a product directory to the :class:`~satimg.product.Product` subclass
that can read it, by matching each registered regex against the directory name.

Products register themselves with the :func:`register` decorator; importing
:mod:`satimg.products` pulls them all in.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable, TypeVar

from satimg.product import Product

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=type[Product])

_REGISTRY: list[tuple[re.Pattern[str], type[Product]]] = []


class UnknownProductError(ValueError):
    """Raised when no registered product recognises a path."""


def register(pattern: str) -> Callable[[T], T]:
    """Class decorator: associate ``pattern`` (matched against the folder name)
    with a product class."""

    compiled = re.compile(pattern)

    def decorate(cls: T) -> T:
        _REGISTRY.append((compiled, cls))
        return cls

    return decorate


def resolve(path: str) -> type[Product]:
    """Return the product class for ``path`` or raise :class:`UnknownProductError`."""
    name = Path(path).name
    for pattern, cls in _REGISTRY:
        if pattern.search(name):
            logger.debug(
                "resolved %s via pattern %r -> %s", name, pattern.pattern, cls.__name__
            )
            return cls
    raise UnknownProductError(path)


def scene_key(name: str) -> str:
    """The scene a product belongs to: its name (a path, folder or zip name, or
    a connector's search item id) without the parts that change when the scene
    is reprocessed, so every version of one acquisition gets the same key --
    e.g. ``S1A_IW_GRDH_1SDV_..._04368F_4953_COG`` and ``..._04368F_A1B2_COG``
    both give ``S1A_IW_GRDH_1SDV_..._04368F``. Names no product recognises
    (or whose product has no such parts) are returned as they are."""
    name = Path(name).name.removesuffix(".zip")
    for cls in dict.fromkeys(cls for _, cls in _REGISTRY):
        key = cls.scene_key(name)
        if key is not None:
            return key
    return name
