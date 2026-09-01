"""Map a product directory to the :class:`~satimg.product.Product` subclass
that can read it.

Resolution tries, in order:

1. each registered regex against the directory name;
2. each class's optional :meth:`Product.sniff` classmethod (content inspection).

Products register themselves with the :func:`register` decorator; importing
:mod:`satimg.products` pulls them all in.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, TypeVar

from satimg.product import Product

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


def registry() -> list[tuple[str, type[Product]]]:
    """The registered ``(pattern, class)`` pairs, for inspection."""
    return [(p.pattern, c) for p, c in _REGISTRY]


def resolve(path: str) -> type[Product]:
    """Return the product class for ``path`` or raise :class:`UnknownProductError`."""
    name = Path(path).name
    for pattern, cls in _REGISTRY:
        if pattern.search(name):
            return cls
    for _, cls in _REGISTRY:
        sniff = getattr(cls, "sniff", None)
        if callable(sniff) and sniff(path):
            return cls
    raise UnknownProductError(path)
