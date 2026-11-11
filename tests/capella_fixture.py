"""A small synthetic Capella GEC product folder, laid out as the open-data
bucket serves it -- real GECs are hundreds of MB.

``CAPELLA_C13_SP_GEC_HH_..._20260116014119/``: the ``.tif`` (``uint16``,
north-up UTM 11N, with overviews), the ``_extended.json`` (only the fields
satimg reads), and optionally the GEO quick-look PNG. Sea everywhere at
sigma0 -20 dB, one bright 'ship' at :data:`SHIP` at +10 dB, a zero
(no-data) border.
"""

from __future__ import annotations

import datetime
import json

import numpy
from pyproj import Transformer

from tests import orbit_fixture

NAME = "CAPELLA_C13_SP_GEC_HH_20260116014108_20260116014119"
HEIGHT, WIDTH = 256, 200
SPACING = 0.5
X0, Y0 = 368_857.0, 3_739_850.0
EPSG = 32611
SHIP = (100, 120)
SCALE = 0.0009
SEA_DB, SHIP_DB = -20.0, 10.0
#: the imaging geometry at :data:`TARGET` (the grid's middle), for the
#: synthetic orbit: left-looking, looking east-north-east
INCIDENCE, SLANT_RANGE, LOOK = 49.36, 700_000.0, 80.0
CENTRE_TIME = datetime.datetime(2026, 1, 16, 1, 41, 14, 75999)
TARGET = tuple(reversed(Transformer.from_crs(EPSG, 4326, always_xy=True).transform(
    X0 + SPACING * WIDTH / 2, Y0 - SPACING * HEIGHT / 2
)))


def _stamp(seconds: float) -> str:
    return (CENTRE_TIME + datetime.timedelta(seconds=seconds)).isoformat() + "000Z"


def _state_vectors() -> list[dict]:
    times, positions, velocities = orbit_fixture.states(
        *TARGET, look_bearing=LOOK, incidence=INCIDENCE, slant_range=SLANT_RANGE, right_looking=False
    )
    return [
        {"time": _stamp(t), "position": p.tolist(), "velocity": v.tolist()}
        for t, p, v in zip(times, positions, velocities)
    ]


EXTENDED = {
    "product_type": "GEC",
    "collect": {
        "platform": "capella-13",
        "mode": "spotlight",
        "collect_id": "57df49b4-f8fa-4458-b3b2-fbb6f4883da3",
        "start_timestamp": _stamp(-5.5),
        "stop_timestamp": _stamp(5.5),
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
        "state": {"direction": "ascending", "coordinate_system": {"type": "ecef"}, "state_vectors": _state_vectors()},
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
