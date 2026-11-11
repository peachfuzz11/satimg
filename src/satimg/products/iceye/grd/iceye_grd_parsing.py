"""Parsing an ICEYE GRD product.

A product, as ICEYE's open-data bucket lays it out, is a folder named for
the image -- ``ICEYE_4EKGVZ_20251110T124428Z_7004635_X42_SLEDF/`` (scene
id, start time, image id, satellite, imaging mode) -- holding, among the
SLC and other derived products, ``<name>_GRD.tif``: the ground-range
detected image as a cloud-optimised GeoTIFF. The GRD is self-contained:
its ``ICEYE_PROPERTIES`` tag carries ICEYE's full STAC metadata (times,
imaging geometry, ``iceye:calibration_factor``), its GCPs a regular grid
of ground positions, and its band scale / offset the packing of the
amplitude into ``uint16``.
"""

from __future__ import annotations

import datetime
import json
import re

import numpy
import rasterio

from satimg.source.source import Source

#: the folder (or ``_GRD.tif``) name
NAME = re.compile(
    r"^ICEYE_(?P<scene>[0-9A-Z]{6})_(?P<start>\d{8}T\d{6}Z)_(?P<image>\d+)_(?P<sat>X\d+)_(?P<mode>[0-9A-Z]+)"
)


def stem(source: Source) -> str:
    """The product's file-name stem (its folder name, no ``_GRD.tif``)."""
    m = NAME.match(source.name)
    if m is None:
        raise ValueError(f"not an ICEYE product: {source.name}")
    return m.group(0)


def find_grd(source: Source) -> str:
    """The GRD raster's path under ``source``."""
    tif = f"{stem(source)}_GRD.tif"
    if not source.exists(tif):
        raise FileNotFoundError(f"no {tif} under {source.name}")
    return tif


def _time(text: str) -> datetime.datetime:
    """A UTC timestamp as a naive datetime (satimg's convention)."""
    t = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    return t.astimezone(datetime.timezone.utc).replace(tzinfo=None)


def _fill_column(values: numpy.ndarray, rows: numpy.ndarray) -> None:
    """Fill ``values``' NaNs in place, each linearly from the two nearest
    present nodes of its column (extrapolating past the last one)."""
    present = numpy.flatnonzero(~numpy.isnan(values))
    if len(present) < 2:
        raise ValueError("a GCP grid column has fewer than two GCPs")
    for i in numpy.flatnonzero(numpy.isnan(values)):
        a, b = sorted(present[numpy.argsort(numpy.abs(present - i))[:2]])
        t = (rows[i] - rows[a]) / (rows[b] - rows[a])
        values[i] = values[a] + t * (values[b] - values[a])


def gcp_grid(gcps) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """``(rows, cols, lat, lon)`` of GCPs laid on a regular grid -- node row
    and column pixel indices (pixel centres, as ICEYE's GCPs are), and the
    ``(len(rows), len(cols))`` latitude / longitude at each node.

    A few GRDs lack a handful of nodes at an image corner, beyond the imaged
    area; those are filled in from their column's neighbours."""
    rows = numpy.array(sorted({g.row for g in gcps}))
    cols = numpy.array(sorted({g.col for g in gcps}))
    lat = numpy.full((len(rows), len(cols)), numpy.nan)
    lon = numpy.full_like(lat, numpy.nan)
    i, j = numpy.searchsorted(rows, [g.row for g in gcps]), numpy.searchsorted(cols, [g.col for g in gcps])
    lat[i, j] = [g.y for g in gcps]
    lon[i, j] = [g.x for g in gcps]
    for k in numpy.flatnonzero(numpy.isnan(lat).any(axis=0)):
        _fill_column(lat[:, k], rows)
        _fill_column(lon[:, k], rows)
    return rows, cols, lat, lon


def epoch(props: dict) -> datetime.datetime:
    """The scene centre's imaging time: the middle of the zero-Doppler span
    (a GRD is in zero-Doppler geometry), else the item's ``datetime``."""
    start, end = props.get("iceye:zero_doppler_start_datetime"), props.get("iceye:zero_doppler_end_datetime")
    if start and end:
        t0 = _time(start)
        return t0 + (_time(end) - t0) / 2
    return _time(props["datetime"])


def orbit(props: dict) -> dict | None:
    """The sensor orbit as
    :class:`~satimg.sar_orbit_geometry.SAROrbitGeometry` reads it: ICEYE's
    ECEF ``iceye:orbit_states`` (a few dozen, spanning the acquisition),
    timed from :func:`epoch`; ``None`` without them."""
    states = props.get("iceye:orbit_states")
    if not states or props.get("iceye:coordinate_frame", "ecef").lower() != "ecef":
        return None
    t0 = epoch(props)
    return {
        "epoch": t0.isoformat() + "Z",
        "times": [(_time(s["time"]) - t0).total_seconds() for s in states],
        "positions": [list(map(float, s["position"])) for s in states],
        "velocities": [list(map(float, s["velocity"])) for s in states],
    }


def read_metadata(path: str) -> dict:
    """``{"timestamp", "calibration_factor", "scale", "offset", "shape",
    "nodes", "attrs"}`` from the GRD's header alone -- ``timestamp`` the
    scene centre's time (naive UTC); ``scale`` / ``offset`` unpack a pixel
    value into amplitude, which ``calibration_factor`` turns into sigma
    nought (``sigma0 = cf * (DN * scale + offset)^2``); ``nodes`` the GCP
    grid (:func:`gcp_grid`); ``attrs`` the scene-wide values below (angles in
    degrees, distances in metres; ``orbit`` see :func:`orbit`)."""
    with rasterio.open(path) as src:
        props = json.loads(src.tags()["ICEYE_PROPERTIES"])
        scale, offset = src.scales[0], src.offsets[0]
        shape = (src.height, src.width)
        gcps, _ = src.gcps
    cf = float(props["iceye:calibration_factor"])
    return {
        "timestamp": _time(props["datetime"]),
        "calibration_factor": cf,
        "scale": float(scale),
        "offset": float(offset),
        "shape": shape,
        "nodes": gcp_grid(gcps),
        "attrs": {
            "platform": props.get("platform", "").lower() or None,
            "mode": props.get("sar:instrument_mode"),
            "imaging_mode": props.get("iceye:processing_mode"),
            "polarisations": list(props["sar:polarizations"]),
            "look_direction": props.get("sar:observation_direction"),
            "pass": props.get("sat:orbit_state"),
            "incidence_angle": props.get("view:incidence_angle"),
            "look_angle": props.get("view:off_nadir"),
            "squint_angle": props.get("iceye:squint_angle"),
            "slant_range": props.get("iceye:range"),
            "resolution_range": props.get("sar:resolution_range"),
            "resolution_azimuth": props.get("sar:resolution_azimuth"),
            "looks_range": props.get("sar:looks_range"),
            "looks_azimuth": props.get("sar:looks_azimuth"),
            "calibration_factor": cf,
            "scene_id": props.get("iceye:scene_id"),
            "image_id": props.get("iceye:image_id"),
            "orbit": orbit(props),
        },
    }
