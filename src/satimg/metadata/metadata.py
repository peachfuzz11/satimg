"""A product's named metadata fields plus its scene-wide attributes."""

from __future__ import annotations

from satimg.metadata.field import Field


class Metadata:
    """Named :class:`Field` s plus scalar scene metadata (:attr:`attrs`)."""

    def __init__(self, fields: dict[str, Field], attrs: dict | None = None):
        self._fields: dict[str, Field] = dict(fields)
        self.attrs: dict = dict(attrs or {})

    @property
    def fields(self) -> list[str]:
        return list(self._fields)

    def __getitem__(self, name: str) -> Field:
        return self._fields[name]

    def __contains__(self, name: str) -> bool:
        return name in self._fields

    def __iter__(self):
        return iter(self._fields)

    def get(self, name: str, default=None):
        return self._fields.get(name, default)

    def __getattr__(self, name: str) -> Field:
        # reached only when normal attribute lookup fails
        try:
            return object.__getattribute__(self, "_fields")[name]
        except KeyError:
            raise AttributeError(name) from None

    def at(self, coords) -> dict:
        """``{field: value}`` for every field at ``coords`` (see :meth:`Field.at`)."""
        return {name: field.at(coords) for name, field in self._fields.items()}

    sample = at

    def corners(self) -> dict:
        """``{field: {"top_left": v, ...}}`` for every field over the full product
        grid (see :meth:`Field.corners`)."""
        return {name: field.corners() for name, field in self._fields.items()}

    def __repr__(self) -> str:
        if not self._fields:
            return "Metadata(empty)"
        inner = ", ".join(
            f"{n} ({f.units})" if f.units else n for n, f in self._fields.items()
        )
        return f"Metadata({inner})"
