"""Metadata seen through one patch window."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy
import xarray

from satimg.metadata.metadata import Metadata
from satimg.metadata.field import CORNER_KEYS, corner_center

if TYPE_CHECKING:  # pragma: no cover
    from satimg.geometry import Window


class PatchMeta:
    """Lazy metadata view bound to one patch window; returned by ``Patch.meta``."""

    __slots__ = ("_meta", "_window")

    def __init__(self, metadata: Metadata, window: Window):
        self._meta = metadata
        self._window = window

    def __getitem__(self, name: str) -> xarray.DataArray:
        return self._meta[name].read(self._window)

    def __getattr__(self, name: str) -> xarray.DataArray:
        meta = object.__getattribute__(self, "_meta")
        if name in meta:
            return meta[name].read(object.__getattribute__(self, "_window"))
        raise AttributeError(name)

    def at(self, rc) -> dict:
        """``{field: value}`` at ``rc`` -- ``(row, col)`` measured *inside* the
        patch (a pair, a list of pairs, or an ``(N, 2)`` array)."""
        arr = numpy.asarray(rc, dtype=float)
        single = arr.ndim == 1
        pairs = arr.reshape(-1, 2)
        glob = numpy.column_stack(
            (pairs[:, 0] + self._window.row, pairs[:, 1] + self._window.col)
        )
        out = {name: self._meta[name].at(glob) for name in self._meta.fields}
        if single:
            out = {name: float(numpy.asarray(val)[0]) for name, val in out.items()}
        return out

    def sample(self) -> dict:
        """``{field: value}`` at the patch centre."""
        return self.at((self._window.height / 2, self._window.width / 2))

    def corners(self) -> dict:
        """``{field: {"top_left": v, "top_right": v, "bottom_left": v,
        "bottom_right": v, "center": v}}`` for this patch window."""
        pts = corner_center(self._window.height, self._window.width)
        per = self.at(pts)  # {field: (5,) array}
        return {
            name: {k: float(v) for k, v in zip(CORNER_KEYS, vals)}
            for name, vals in per.items()
        }

    def __repr__(self) -> str:
        return f"PatchMeta({self._meta.fields}, {self._window})"
