"""Shared by the sensor-geometry swath products that ship a latitude /
longitude per pixel (VIIRS, MODIS, Sentinel-3 OLCI): choosing the coarse
node grid a :class:`~satimg.transform.swath_transformer.SwathTransformer` and the per-pixel
metadata fields are kept on, and tracing a footprint.
"""

from __future__ import annotations

import numpy

from satimg.transform.swath_transformer import EARTH_RADIUS_M, unit_vectors


def node_columns(lat: numpy.ndarray, lon: numpy.ndarray, tolerance_m: float) -> numpy.ndarray:
    """The fewest columns (greedily) between which linear interpolation of
    ``lat`` / ``lon`` (rows, all columns) stays within ``tolerance_m`` metres
    of the truth on every row given. Ground sampling across a swath changes
    in steps (VIIRS's aggregation zones, OLCI's five cameras), so evenly
    spaced columns would need several times more nodes for the same
    accuracy. Transpose both to choose rows instead."""
    width = lat.shape[1]
    truth = unit_vectors(lat, lon)  # (rows, width, 3)
    nodes = numpy.array([0, width - 1])
    every = numpy.arange(width)
    while True:
        k = numpy.clip(numpy.searchsorted(nodes, every, side="right") - 1, 0, len(nodes) - 2)
        t = (every - nodes[k]) / (nodes[k + 1] - nodes[k])
        guess = truth[:, nodes[k]] * (1 - t)[:, None] + truth[:, nodes[k + 1]] * t[:, None]
        guess /= numpy.linalg.norm(guess, axis=-1, keepdims=True)
        error = numpy.nan_to_num(numpy.linalg.norm(guess - truth, axis=-1) * EARTH_RADIUS_M).max(axis=0)
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
