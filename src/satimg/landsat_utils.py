"""Landsat Collection-2 Level-1 parsing: ``MTL.json`` (acquisition/projection
attrs) and the per-pixel angle rasters (``SZA``/``SAA``/``VZA``/``VAA.TIF``).

Pure functions taking a :class:`~satimg.source.Source` (or a real filesystem
path, for the angle rasters -- read with ``rasterio``, so raster-only like
:meth:`~satimg.product.Product._open_raw`) and returning plain dicts/arrays/a
:class:`~satimg.transform.Transformer` -- no dependency on
:class:`~satimg.product.Product`, so they're testable independently of a full
product open. See :class:`~satimg.products.landsat.LandsatProduct` for the
product-facing entry points that call these.
"""

from __future__ import annotations

import datetime
import json

import numpy
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from satimg.source import Source
from satimg.transform import Transformer

#: band order of :attr:`~satimg.products.landsat.LandsatProduct.raw`.
BANDS = ("coastal", "blue", "green", "red", "nir08", "swir16", "swir22", "pan",
         "cirrus", "lwir11", "lwir12")
PAN = BANDS.index("pan")
RGB = (BANDS.index("red"), BANDS.index("green"), BANDS.index("blue"))

#: per-pixel angle rasters (int16, hundredths of a degree) -> metadata field name.
ANGLE_FILES = {
    "sun_zenith": "SZA.TIF",
    "sun_azimuth": "SAA.TIF",
    "view_zenith": "VZA.TIF",
    "view_azimuth": "VAA.TIF",
}
ANGLE_SCALE = 0.01


def parse_mtl(source: Source) -> dict:
    """``MTL.json``'s image attrs, acquisition timestamp, footprint, and
    :class:`~satimg.transform.Transformer` (UTM zone/hemisphere resolved from
    ``PROJECTION_ATTRIBUTES``, since Landsat ships no CRS/affine transform of
    its own the way an embedded GCP list or ``MTD_TL.xml`` geocoding block
    would)."""
    with source.open("MTL.json") as f:
        mtl = json.load(f)["LANDSAT_METADATA_FILE"]

    attrs = mtl["IMAGE_ATTRIBUTES"]
    timestamp = datetime.datetime.fromisoformat(
        f"{attrs['DATE_ACQUIRED']}T{attrs['SCENE_CENTER_TIME']}"
    )
    proj = mtl["PROJECTION_ATTRIBUTES"]
    ring = [
        (float(proj[f"CORNER_{c}_LON_PRODUCT"]), float(proj[f"CORNER_{c}_LAT_PRODUCT"]))
        for c in ("UL", "UR", "LR", "LL", "UL")
    ]
    footprint = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }
    if proj.get("MAP_PROJECTION") != "UTM":
        raise NotImplementedError(
            f"unsupported MAP_PROJECTION {proj.get('MAP_PROJECTION')!r}"
        )
    zone = int(proj["UTM_ZONE"])
    epsg = 32600 + zone if float(proj["CORNER_UL_LAT_PRODUCT"]) >= 0 else 32700 + zone
    gsd = float(proj["GRID_CELL_SIZE_PANCHROMATIC"])
    # MTL's CORNER_UL_PROJECTION_*_PRODUCT is the UL pixel's *centre*; a
    # geotransform origin is that pixel's corner, half a pixel out.
    ul_x = float(proj["CORNER_UL_PROJECTION_X_PRODUCT"]) - gsd / 2
    ul_y = float(proj["CORNER_UL_PROJECTION_Y_PRODUCT"]) + gsd / 2
    transform = Affine(gsd, 0, ul_x, 0, -gsd, ul_y)
    transformer = Transformer(transform, CRS.from_epsg(epsg))

    return {
        "image_attrs": attrs,
        "timestamp": timestamp,
        "footprint": footprint,
        "transformer": transformer,
    }


def read_angle_grid(
    path: str, pan_transform: Affine
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """One angle-raster file's decimated ``(rows, cols, values)``, ``rows``/
    ``cols`` expressed in the pan-band's own pixel coordinates, for
    :meth:`~satimg.product.Product._field`."""
    with rasterio.open(path) as src:
        stride = max(1, max(src.height, src.width) // 256)
        out_h, out_w = src.height // stride, src.width // stride
        values = src.read(1, out_shape=(out_h, out_w)).astype(float) * ANGLE_SCALE
        grid_t = src.transform * Affine.scale(src.width / out_w, src.height / out_h)
    # angle-grid sample centres expressed in pan-grid pixel coordinates
    cols = (grid_t.c + (numpy.arange(out_w) + 0.5) * grid_t.a - pan_transform.c) / pan_transform.a - 0.5
    rows = (grid_t.f + (numpy.arange(out_h) + 0.5) * grid_t.e - pan_transform.f) / pan_transform.e - 0.5
    return rows, cols, values
