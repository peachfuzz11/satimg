"""A small synthetic Capella GEC product folder, laid out as the open-data
bucket serves it -- real GECs are hundreds of MB.

``CAPELLA_C13_SP_GEC_HH_..._20260116014119/``: the ``.tif`` (``uint16``,
north-up UTM 11N, with overviews), the ``_extended.json`` (only the fields
satimg reads), and optionally the GEO quick-look PNG. Sea everywhere at
sigma0 -20 dB, one bright 'ship' at :data:`SHIP` at +10 dB, a zero
(no-data) border.
"""

from __future__ import annotations

import json

import numpy

NAME = "CAPELLA_C13_SP_GEC_HH_20260116014108_20260116014119"
HEIGHT, WIDTH = 256, 200
SPACING = 0.5
X0, Y0 = 368_857.0, 3_739_850.0
EPSG = 32611
SHIP = (100, 120)
SCALE = 0.0009
SEA_DB, SHIP_DB = -20.0, 10.0

EXTENDED = {
    "product_type": "GEC",
    "collect": {
        "platform": "capella-13",
        "mode": "spotlight",
        "collect_id": "57df49b4-f8fa-4458-b3b2-fbb6f4883da3",
        "image": {
            "scale_factor": SCALE,
            "radiometry": "sigma_nought",
            "pixel_spacing_row": SPACING,
            "pixel_spacing_column": SPACING,
            "ground_range_resolution": 0.42,
            "ground_azimuth_resolution": 0.56,
            "enl": 3.0,
            "center_pixel": {
                "incidence_angle": 49.36,
                "look_angle": 43.85,
                "squint_angle": 0.0,
                "center_time": "2026-01-16T01:41:14.075999123Z",
            },
        },
        "radar": {"pointing": "left", "transmit_polarization": "H", "receive_polarization": "H"},
        "state": {"direction": "ascending"},
    },
}


def dn(db: float) -> int:
    """The pixel value whose sigma0 is ``db`` dB."""
    return round(10 ** (db / 20) / SCALE)


def image() -> numpy.ndarray:
    data = numpy.full((HEIGHT, WIDTH), dn(SEA_DB), dtype="uint16")
    data[SHIP[0] - 3: SHIP[0] + 4, SHIP[1] - 1: SHIP[1] + 2] = dn(SHIP_DB)
    data[:8], data[-8:], data[:, :8], data[:, -8:] = 0, 0, 0, 0
    return data


def build(parent, *, thumbnail: bool = False):
    """Write the product folder under ``parent``; returns its path."""
    import PIL.Image
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin

    folder = parent / NAME
    folder.mkdir()
    with rasterio.open(
        folder / f"{NAME}.tif", "w", driver="GTiff", width=WIDTH, height=HEIGHT, count=1,
        dtype="uint16", crs=f"EPSG:{EPSG}", transform=from_origin(X0, Y0, SPACING, SPACING),
        nodata=0, tiled=True, blockxsize=128, blockysize=128,
    ) as dst:
        dst.write(image(), 1)
        dst.build_overviews([2, 4], Resampling.average)
    (folder / f"{NAME}_extended.json").write_text(json.dumps(EXTENDED))
    if thumbnail:
        thumb = NAME.replace("_GEC_", "_GEO_") + "_thumb.png"
        PIL.Image.new("L", (64, 32), 77).save(folder / thumb)
    return folder
