"""Sentinel-2 L1C SAFE parsing: ``MTD_MSIL1C.xml`` (band/tile inventory) and
``MTD_TL.xml`` (acquisition time, sun/viewing angle grids, per-resolution
geocoding).

Pure functions taking a :class:`~satimg.source.Source` and returning plain
dicts/arrays -- no dependency on :class:`~satimg.product.Product`, so they're
testable independently of a full product open. See
:class:`~satimg.products.sentinel2.Sentinel2L1CProduct` for the
product-facing entry point that calls these.
"""

from __future__ import annotations

import datetime
import logging
import os
import re
import warnings
from xml.etree import ElementTree

import numpy
from rasterio.crs import CRS
from rasterio.transform import Affine

from satimg.metadata import fill_nan_nearest, regular_axis
from satimg.source import Source

logger = logging.getLogger(__name__)

#: spacing of the MTD_TL.xml angle grids, metres.
ANGLE_STEP_M = 5000.0

#: Sentinel-2 L1C per-band native ground sample distance, metres -- picks the
#: ``Tile_Geocoding/Geoposition`` (and matching ``Size``) that
#: ``band_names[1]`` (the :func:`~satimg.readers.merge_bands` ``match=``
#: target) was actually shot at, so the zip-native transformer/grid-shape
#: agree exactly with opening that band's own JP2.
BAND_GSD_M = {
    "B01": 60, "B02": 10, "B03": 10, "B04": 10, "B05": 20, "B06": 20,
    "B07": 20, "B08": 10, "B8A": 20, "B09": 60, "B10": 60, "B11": 20, "B12": 20,
}


def pad_band(name: str) -> str:
    """``B1`` -> ``B01``; leaves ``B8A`` / ``B10`` .. ``B12`` unchanged."""
    return re.sub(r"^B(\d)$", r"B0\1", name)


def parse_mtd(source: Source, path: str) -> dict:
    """``path`` (a real filesystem path, unlike ``source``) is only used to
    build ``reflectance``/``tci``'s raster paths -- raster-only, so still a
    plain path even when ``source`` is zip-native (where those two entries
    are then unused, since ``raw``/``visual`` are guarded off)."""
    mtd = source.find_file("MTD_MSIL1C.xml")
    if mtd is None:
        raise FileNotFoundError(f"no MTD_MSIL1C.xml under {source.name}")
    with source.open(mtd) as f:
        root = ElementTree.parse(f).getroot()

    physical, wavelengths = {}, {}
    for info in root.findall(".//Spectral_Information"):
        bid = int(info.get("bandId"))
        physical[bid] = info.get("physicalBand")
        central = info.find(".//CENTRAL")
        if central is not None:
            wavelengths[bid] = float(central.text)

    reflectance, tci = [], None
    for band_id, elem in enumerate(root.iter("IMAGE_FILE")):
        file_path = os.path.join(path, elem.text) + ".jp2"
        if physical.get(band_id) is None:
            tci = file_path
        else:
            reflectance.append(file_path)

    n = len(reflectance)
    band_names = [pad_band(physical[i]) for i in range(n)]
    band_wavelengths = [wavelengths.get(i, numpy.nan) for i in range(n)]

    ext = [float(v) for v in root.find(".//EXT_POS_LIST").text.split()]
    ring = [(ext[i + 1], ext[i]) for i in range(0, len(ext), 2)]
    return {
        "reflectance": reflectance,
        "band_names": band_names,
        "wavelengths": band_wavelengths,
        "tci": tci,
        "footprint": {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        },
    }


def _angle_grid(node: ElementTree.Element) -> numpy.ndarray:
    """The ``Values_List`` of a Zenith/Azimuth node as a 2-D array (``NaN`` ok)."""
    rows = node.find("Values_List").findall("VALUES")
    return numpy.array([[float(v) for v in row.text.split()] for row in rows])


def _mean_grids(grids: list[numpy.ndarray], *, circular: bool) -> numpy.ndarray:
    """NaN-aware mean of the per-detector grids; ``circular`` for azimuths. Cells
    that no detector covers (tile corners) are then nearest-filled."""
    stack = numpy.stack(grids)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices
        if not circular:
            merged = numpy.nanmean(stack, axis=0)
        else:
            rad = numpy.deg2rad(stack)
            merged = numpy.rad2deg(
                numpy.arctan2(numpy.nanmean(numpy.sin(rad), 0), numpy.nanmean(numpy.cos(rad), 0))
            ) % 360.0
    return fill_nan_nearest(merged)


def _view_grid(
    grids: list[numpy.ndarray], shape: tuple[int, int], fallback: float, *, circular: bool, name: str
) -> numpy.ndarray:
    """The merged per-detector viewing grid on the tile's ``shape`` angle grid.

    A detector grid of any other shape can't be placed: ``Values_List`` carries
    no offset, so its cells could sit anywhere on the tile. Such grids (seen on
    products that only clip a tile's edge) are dropped rather than guessed at,
    and with none left the grid is the constant ``fallback`` -- the band-mean
    viewing angle -- instead."""
    placeable = [g for g in grids if g.shape == shape]
    if len(placeable) < len(grids):
        logger.warning(
            "%s: dropping %d of %d detector grids whose shape is not the tile's %s",
            name, len(grids) - len(placeable), len(grids), shape,
        )
    if not placeable:
        return numpy.full(shape, fallback)
    return _mean_grids(placeable, circular=circular)


def _mean_view_angles(root: ElementTree.Element) -> tuple[float, float]:
    """``(zenith, azimuth)`` averaged over ``Mean_Viewing_Incidence_Angle_List``'s
    bands (azimuth circularly); ``NaN`` where the list is missing."""
    means = root.findall(".//Mean_Viewing_Incidence_Angle")
    if not means:
        return float("nan"), float("nan")
    zenith = numpy.array([float(m.find("ZENITH_ANGLE").text) for m in means])
    azimuth = numpy.deg2rad([float(m.find("AZIMUTH_ANGLE").text) for m in means])
    mean_azimuth = numpy.rad2deg(numpy.arctan2(numpy.sin(azimuth).mean(), numpy.cos(azimuth).mean())) % 360.0
    return float(zenith.mean()), float(mean_azimuth)


def parse_tl(source: Source) -> dict:
    """Acquisition time, the sun / viewing angle grids, and the
    ``Tile_Geocoding`` CRS + per-resolution affine transform / pixel-grid
    shape (for the zip-native transformer/metadata-grid-shape, in place of
    opening a band's own JP2), all from ``MTD_TL.xml``."""
    tl = source.find_file("MTD_TL.xml")
    if tl is None:
        raise FileNotFoundError(f"no MTD_TL.xml under {source.name}")
    with source.open(tl) as f:
        root = ElementTree.parse(f).getroot()

    timestamp = datetime.datetime.strptime(
        root.find(".//SENSING_TIME").text, "%Y-%m-%dT%H:%M:%S.%fZ"
    )

    geocoding = root.find(".//Tile_Geocoding")
    crs = CRS.from_string(geocoding.find("HORIZONTAL_CS_CODE").text)
    transforms = {
        int(g.get("resolution")): Affine(
            float(g.find("XDIM").text), 0, float(g.find("ULX").text),
            0, float(g.find("YDIM").text), float(g.find("ULY").text),
        )
        for g in geocoding.findall("Geoposition")
    }
    pixel_shapes = {
        int(s.get("resolution")): (int(s.find("NROWS").text), int(s.find("NCOLS").text))
        for s in geocoding.findall("Size")
    }

    pixel_m = min(transforms)
    step_px = ANGLE_STEP_M / pixel_m

    sun = root.find(".//Sun_Angles_Grid")
    grids = {
        "sun_zenith": _angle_grid(sun.find("Zenith")),
        "sun_azimuth": _angle_grid(sun.find("Azimuth")),
    }
    # the sun grid always spans the whole tile; every field shares its axes
    shape = grids["sun_zenith"].shape
    view = root.findall(".//Viewing_Incidence_Angles_Grids")
    mean_view_zenith, mean_view_azimuth = _mean_view_angles(root)
    grids["view_zenith"] = _view_grid(
        [_angle_grid(g.find("Zenith")) for g in view], shape, mean_view_zenith,
        circular=False, name="view_zenith",
    )
    grids["view_azimuth"] = _view_grid(
        [_angle_grid(g.find("Azimuth")) for g in view], shape, mean_view_azimuth,
        circular=True, name="view_azimuth",
    )

    axes = (regular_axis(shape[0], step_px), regular_axis(shape[1], step_px))

    mean_sun = root.find(".//Mean_Sun_Angle")
    attrs = {
        "mean_sun_zenith": float(mean_sun.find("ZENITH_ANGLE").text),
        "mean_sun_azimuth": float(mean_sun.find("AZIMUTH_ANGLE").text),
    }
    return {
        "timestamp": timestamp, "axes": axes, "grids": grids, "attrs": attrs,
        "crs": crs, "transforms": transforms, "pixel_shapes": pixel_shapes,
    }
