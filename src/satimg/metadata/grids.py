"""Coarse metadata grids: interpolating them, and building them from what
the products ship."""

from __future__ import annotations

import numpy


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
    # a single-row/-column axis gives r1 == r0 (c1 == c0): divide by 1 there
    # rather than 0, so numpy.where's untaken branch doesn't warn
    tr = numpy.where(r1 > r0, (qr - r0) / numpy.where(r1 > r0, r1 - r0, 1.0), 0.0)
    tc = numpy.where(c1 > c0, (qc - c0) / numpy.where(c1 > c0, c1 - c0, 1.0), 0.0)

    top = grid[i, j] * (1 - tc) + grid[i, j + 1] * tc
    bot = grid[i + 1, j] * (1 - tc) + grid[i + 1, j + 1] * tc
    return top * (1 - tr) + bot * tr


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
