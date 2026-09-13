"""Array/band helpers shared by the product readers and by windowed reads.

:func:`as_band_yx` / :func:`label_bands` normalise a raster's dims and band
labels; :func:`read_window` cuts a lazy, zero-padded sub-array for a
:class:`~satimg.geometry.Window` out of any ``(band, y, x)`` or bare ``(y, x)``
array. Windowed *iteration* itself lives on :class:`~satimg.product.Product`
(``patches()`` / ``patches_at()``), which is the only place a
:class:`~satimg.product.Patch` is built.
"""

from __future__ import annotations

import xarray

from satimg.geometry import Window


def as_band_yx(data: xarray.DataArray) -> xarray.DataArray:
    """Return ``data`` with dims exactly ``(band, y, x)``, adding a singleton
    ``band`` axis if the array is 2-D."""
    if "band" not in data.dims:
        data = data.expand_dims("band")
    missing = {"band", "y", "x"} - set(data.dims)
    if missing:
        raise ValueError(f"expected band/y/x dims, missing {missing}: got {data.dims}")
    return data.transpose("band", "y", "x", *[d for d in data.dims if d not in ("band", "y", "x")])


def label_bands(da: xarray.DataArray, names) -> xarray.DataArray:
    """Attach ``names`` as the ``band`` coordinate of ``da`` (a ``(band, y, x)``
    array). ``len(names)`` must equal the band count."""
    names = list(names)
    if len(names) != da.sizes["band"]:
        raise ValueError(
            f"got {len(names)} band names for {da.sizes['band']} bands: {names}"
        )
    return da.assign_coords(band=names)


def read_window(da: xarray.DataArray, window: Window) -> xarray.DataArray:
    """Lazy sub-array of ``da`` for ``window``.

    The result always has the window's exact ``width``/``height`` -- if the
    window overhangs the array it is zero-padded. ``col``/``row`` are stored in
    ``.attrs`` so the origin survives a round-trip through numpy. Only the ``x``
    and ``y`` axes are touched, so this works for ``(band, y, x)`` and bare
    ``(y, x)`` arrays alike.
    """
    w, h = int(da.sizes["x"]), int(da.sizes["y"])
    inner = window.clip(w, h)
    sub = da.isel(**inner.isel())
    pad_x = (inner.col - window.col, window.col_end - inner.col_end)
    pad_y = (inner.row - window.row, window.row_end - inner.row_end)
    if any(pad_x) or any(pad_y):
        sub = sub.pad(x=pad_x, y=pad_y, constant_values=0)
    return sub.assign_attrs(col=window.col, row=window.row)
