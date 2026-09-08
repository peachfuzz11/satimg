"""Per-pixel ancillary metadata (incidence angle, sun/view angles, ...).

Every sensor ships geometry that varies across the scene but is stored on a
*coarse* grid -- Sentinel-1's ``geolocationGridPoint`` list, Sentinel-2's 23x23
angle grids, Landsat's decimated angle rasters. :class:`Metadata` exposes those as
named :class:`Field` s that interpolate to any pixel, the same way
:class:`~satimg.transform.Transformer` converts coordinates::

    m = product.metadata
    m.fields                       # ['incidence_angle', 'slant_range_time', ...]
    m.incidence_angle.at((row, col))   # -> float, bilinear on the coarse grid
    m.sample((row, col))           # -> {field: value} for every field
    m.corners()                    # -> {field: {'top_left': ..., ..., 'center': ...}}
    m.attrs                        # scalar scene-level metadata

Each field also materialises as a lazy full-grid ``(y, x)``
:class:`xarray.DataArray` (``m.incidence_angle.grid``), which is what backs
``patch.meta`` when walking a product in windows::

    for patch in product.patches(512):
        patch.meta.incidence_angle     # lazy (h, w) DataArray for this window
        patch.meta.at((y, x))          # -> {field: value} at a patch pixel
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import numpy
import xarray

from satimg import tiling
from satimg.tiling import read_window
from satimg.transform import _as_n2

if TYPE_CHECKING:  # pragma: no cover
    from satimg.geometry import Window
    from satimg.transform import Transformer


def bilinear(
    rows: numpy.ndarray,
    cols: numpy.ndarray,
    grid: numpy.ndarray,
    qr: numpy.ndarray,
    qc: numpy.ndarray,
) -> numpy.ndarray:
    """Bilinear sample of ``grid`` (defined on the rectilinear axes ``rows`` x
    ``cols``) at query positions ``qr`` / ``qc``.

    ``rows`` and ``cols`` are 1-D and ascending but need not be evenly spaced.
    Queries outside the grid clamp to the edge; a ``NaN`` corner yields ``NaN``.
    ``qr`` / ``qc`` may be scalars or any broadcastable arrays; the result takes
    their broadcast shape.
    """
    rows = numpy.asarray(rows, dtype=float)
    cols = numpy.asarray(cols, dtype=float)
    grid = numpy.asarray(grid, dtype=float)
    qr = numpy.clip(numpy.asarray(qr, dtype=float), rows[0], rows[-1])
    qc = numpy.clip(numpy.asarray(qc, dtype=float), cols[0], cols[-1])

    i = numpy.clip(numpy.searchsorted(rows, qr, side="right") - 1, 0, len(rows) - 2)
    j = numpy.clip(numpy.searchsorted(cols, qc, side="right") - 1, 0, len(cols) - 2)
    r0, r1 = rows[i], rows[i + 1]
    c0, c1 = cols[j], cols[j + 1]
    tr = numpy.where(r1 > r0, (qr - r0) / (r1 - r0), 0.0)
    tc = numpy.where(c1 > c0, (qc - c0) / (c1 - c0), 0.0)

    top = grid[i, j] * (1 - tc) + grid[i, j + 1] * tc
    bot = grid[i + 1, j] * (1 - tc) + grid[i + 1, j + 1] * tc
    return top * (1 - tr) + bot * tr


#: keys of the dict returned by every ``corners()``, in the order they map to
#: :func:`_corner_center`.
CORNER_KEYS = ("top_left", "top_right", "bottom_left", "bottom_right", "center")


def _corner_center(height: float, width: float) -> list[tuple[float, float]]:
    """The five ``(row, col)`` points for a ``height`` x ``width`` grid: the four
    corner pixels, then the centre (matching :meth:`PatchMeta.sample`)."""
    h, w = height - 1, width - 1
    return [(0.0, 0.0), (0.0, w), (h, 0.0), (h, w), (height / 2, width / 2)]


class Field:
    """One scalar quantity sampled on a coarse ``(rows, cols, values)`` grid given
    in the product's own pixel coordinates."""

    def __init__(
        self,
        rows,
        cols,
        values,
        *,
        name: str,
        units: str = "",
        transform: "Transformer | None" = None,
        shape: tuple[int, int] | None = None,
    ):
        self._rows = numpy.asarray(rows, dtype=float)
        self._cols = numpy.asarray(cols, dtype=float)
        self._values = numpy.asarray(values, dtype=float)
        if self._values.shape != (len(self._rows), len(self._cols)):
            raise ValueError(
                f"values {self._values.shape} do not match grid "
                f"{(len(self._rows), len(self._cols))}"
            )
        self.name = name
        self.units = units
        self._transform = transform
        self._shape = shape  # (height, width) of the full product grid

    # -- point lookup ---------------------------------------------------
    def at(self, coords):
        """Value(s) at ``coords`` -- a ``(row, col)`` pair, a list of pairs, or an
        ``(N, 2)`` array. A single pair returns a ``float``; many return ``(N,)``.
        """
        rc = _as_n2(coords)
        out = numpy.asarray(bilinear(self._rows, self._cols, self._values, rc[:, 0], rc[:, 1]))
        return float(out[0]) if numpy.ndim(coords) == 1 else out

    def corners(self) -> dict:
        """``{"top_left": v, "top_right": v, "bottom_left": v, "bottom_right": v,
        "center": v}`` -- the field at the four corners and centre of the full
        product grid."""
        if self._shape is None:
            raise AttributeError(f"field {self.name!r} has no product grid attached")
        vals = self.at(_corner_center(*self._shape))
        return {k: float(v) for k, v in zip(CORNER_KEYS, vals)}

    # -- full grid ----------------------------------------------------
    @cached_property
    def grid(self) -> xarray.DataArray:
        """The field bilinearly upsampled to the whole product grid, lazy
        ``(y, x)``."""
        if self._shape is None:
            raise AttributeError(f"field {self.name!r} has no product grid attached")
        import dask.array as darray

        height, width = self._shape
        rows, cols, values = self._rows, self._cols, self._values
        block = darray.blockwise(
            lambda br, bc: bilinear(rows, cols, values, br[:, None], bc[None, :]),
            "yx",
            darray.arange(height, chunks=2048),
            "y",
            darray.arange(width, chunks=2048),
            "x",
            dtype=float,
        )
        return xarray.DataArray(block, dims=("y", "x"))

    def read(self, window: "Window") -> xarray.DataArray:
        """Lazy ``(y, x)`` sub-array for ``window``."""
        return read_window(self.grid, window)

    def patches(self, *args, **kwargs):
        """Walk :attr:`grid` in windows. See :func:`satimg.tiling.patches`."""
        return tiling.patches(self.grid, *args, transformer=self._transform, **kwargs)

    def patches_at(self, *args, **kwargs):
        """Walk :attr:`grid` at given points. See :func:`satimg.tiling.patches_at`."""
        return tiling.patches_at(self.grid, *args, transformer=self._transform, **kwargs)

    def __repr__(self) -> str:
        u = f", units={self.units!r}" if self.units else ""
        return (
            f"Field(name={self.name!r}{u}, "
            f"grid={len(self._rows)}x{len(self._cols)})"
        )


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


class PatchMeta:
    """Lazy metadata view bound to one patch window; returned by ``Patch.meta``."""

    __slots__ = ("_meta", "_window")

    def __init__(self, metadata: Metadata, window: "Window"):
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
        pts = _corner_center(self._window.height, self._window.width)
        per = self.at(pts)  # {field: (5,) array}
        return {
            name: {k: float(v) for k, v in zip(CORNER_KEYS, vals)}
            for name, vals in per.items()
        }

    def __repr__(self) -> str:
        return f"PatchMeta({self._meta.fields}, {self._window})"


# -- construction helpers (used by the product readers) ------------------


def regular_axis(n: int, step_px: float) -> numpy.ndarray:
    """Pixel coordinates of ``n`` grid lines spaced ``step_px`` apart from 0."""
    return numpy.arange(n, dtype=float) * step_px


def grid_from_points(points: list[dict], keys: tuple[str, ...]) -> dict:
    """Turn an irregular list of sample points into rectilinear grids.

    Each ``point`` is a dict with integer ``"row"`` / ``"col"`` plus one entry per
    ``key``. Returns ``{"rows": ..., "cols": ..., <key>: (R, C) array}`` with
    ``NaN`` where a ``(row, col)`` cell is absent.
    """
    rows = numpy.array(sorted({p["row"] for p in points}), dtype=float)
    cols = numpy.array(sorted({p["col"] for p in points}), dtype=float)
    ri = {v: i for i, v in enumerate(rows)}
    ci = {v: i for i, v in enumerate(cols)}
    out = {key: numpy.full((len(rows), len(cols)), numpy.nan) for key in keys}
    for p in points:
        i, j = ri[float(p["row"])], ci[float(p["col"])]
        for key in keys:
            out[key][i, j] = p[key]
    out["rows"], out["cols"] = rows, cols
    return out


def fill_nan_nearest(grid: numpy.ndarray) -> numpy.ndarray:
    """Replace ``NaN`` cells with the nearest finite value by iterative 4-neighbour
    dilation (no scipy). Used for the merged Sentinel-2 viewing-angle grid, whose
    tile-corner cells no detector covers."""
    out = numpy.array(grid, dtype=float)
    if not numpy.isnan(out).any():
        return out
    while numpy.isnan(out).any():
        holes = numpy.isnan(out)
        filled = out.copy()
        for axis in (0, 1):
            for shift in (1, -1):
                nb = numpy.roll(out, shift, axis=axis)
                take = holes & ~numpy.isnan(nb)
                filled[take] = nb[take]
        if numpy.array_equal(numpy.isnan(filled), numpy.isnan(out)):
            break  # nothing left reachable
        out = filled
    return out
