"""Shared machinery for sensor-geometry swaths that ship a latitude /
longitude per pixel (VIIRS, MODIS, Sentinel-3 OLCI): choosing the coarse node
grid a :class:`~satimg.transform.SwathTransformer` and the per-pixel metadata
fields are kept on, and reading scaled HDF5 / netCDF4 variables.
"""

from __future__ import annotations

import numpy

EARTH_RADIUS_M = 6_371_000.0


def unit_vectors(lat, lon) -> numpy.ndarray:
    """``(3, ...)`` earth-centred unit vectors of ``lat`` / ``lon`` degrees."""
    lat, lon = numpy.deg2rad(lat), numpy.deg2rad(lon)
    return numpy.stack([numpy.cos(lat) * numpy.cos(lon), numpy.cos(lat) * numpy.sin(lon), numpy.sin(lat)])


def node_columns(lat: numpy.ndarray, lon: numpy.ndarray, tolerance_m: float) -> numpy.ndarray:
    """The fewest columns (greedily) between which linear interpolation of
    ``lat`` / ``lon`` (rows, all columns) stays within ``tolerance_m`` metres
    of the truth on every row given. Ground sampling across a swath changes
    in steps (VIIRS's aggregation zones, OLCI's five cameras), so evenly
    spaced columns would need several times more nodes for the same
    accuracy. Transpose both to choose rows instead."""
    width = lat.shape[1]
    truth = unit_vectors(lat, lon)  # (3, rows, width)
    nodes = numpy.array([0, width - 1])
    every = numpy.arange(width)
    while True:
        k = numpy.clip(numpy.searchsorted(nodes, every, side="right") - 1, 0, len(nodes) - 2)
        t = (every - nodes[k]) / (nodes[k + 1] - nodes[k])
        guess = truth[:, :, nodes[k]] * (1 - t) + truth[:, :, nodes[k + 1]] * t
        guess /= numpy.linalg.norm(guess, axis=0)
        error = numpy.nan_to_num(numpy.linalg.norm(guess - truth, axis=0) * EARTH_RADIUS_M).max(axis=0)
        bad = numpy.flatnonzero(error > tolerance_m)
        if not len(bad):
            return nodes
        # the worst column in every interval still out of tolerance
        order = bad[numpy.lexsort((-error[bad], k[bad]))]
        _, first = numpy.unique(k[order], return_index=True)
        nodes = numpy.union1d(nodes, order[first])


def scan_edge_rows(height: int, scan_rows: int) -> numpy.ndarray:
    """The first and last row of every ``scan_rows``-row scan: within a
    whisk-broom scan, geolocation is linear in row, so these are the only
    rows a node grid needs."""
    starts = numpy.arange(0, height, scan_rows)
    return numpy.unique(numpy.r_[starts, numpy.minimum(starts + scan_rows - 1, height - 1)])


def read_scaled(dataset, fill_below: float | None = None, key=()) -> numpy.ndarray:
    """An HDF5 / netCDF4 variable -- or just its ``key`` selection (rows,
    slices: anything ``h5py`` indexes by), so a full-resolution variable
    need not be read whole -- as ``float``, its CF ``_FillValue`` (and
    anything at or below ``fill_below``) as ``NaN``, ``scale_factor`` /
    ``add_offset`` applied -- ``h5py`` applies none of them itself."""
    raw = dataset[key]
    data = raw.astype(float)
    attrs = dataset.attrs
    if "_FillValue" in attrs:
        data[raw == numpy.asarray(attrs["_FillValue"]).ravel()[0]] = numpy.nan
    if fill_below is not None:
        data[data <= fill_below] = numpy.nan
    if "scale_factor" in attrs:
        data *= float(numpy.asarray(attrs["scale_factor"]).ravel()[0])
    if "add_offset" in attrs:
        data += float(numpy.asarray(attrs["add_offset"]).ravel()[0])
    return data


def find_dataset(h5, name: str) -> str:
    """The full path of the first dataset called ``name`` anywhere in
    ``h5`` -- for products whose group layout varies (MODIS C7's
    HDF-EOS5 swath groups)."""
    import h5py

    found = []

    def visit(path, obj):
        if isinstance(obj, h5py.Dataset) and path.rsplit("/", 1)[-1] == name:
            found.append(path)
            return True
        return None

    h5.visititems(visit)
    if not found:
        raise KeyError(f"no dataset {name!r} in {h5.filename}")
    return "/" + found[0]


def outline(lat: numpy.ndarray, lon: numpy.ndarray, per_side: int = 8) -> dict:
    """A GeoJSON footprint traced round the swath's edge pixels -- for
    products that ship no footprint polygon of their own."""
    height, width = lat.shape
    rows = numpy.linspace(0, height - 1, per_side + 1).round().astype(int)
    cols = numpy.linspace(0, width - 1, per_side + 1).round().astype(int)
    edge = (
        [(0, c) for c in cols] + [(r, width - 1) for r in rows[1:]]
        + [(height - 1, c) for c in cols[::-1][1:]] + [(r, 0) for r in rows[::-1][1:]]
    )
    ring = [(float(lon[r, c]), float(lat[r, c])) for r, c in edge if numpy.isfinite(lat[r, c])]
    return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]}}
