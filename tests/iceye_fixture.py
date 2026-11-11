"""A small synthetic ICEYE GRD product folder, laid out as the open-data
bucket serves it -- real GRDs are hundreds of MB.

``ICEYE_4EKGVZ_20251110T124428Z_7004635_X42_SLEDF/``: the ``_GRD.tif``
(``uint16``, no CRS or transform but a 4 x 4 GCP grid, the amplitude packed
by a band scale / offset, ICEYE's metadata in its ``ICEYE_PROPERTIES`` tag,
with overviews). Sea everywhere at sigma0 -20 dB, one bright 'ship' at
:data:`SHIP` at +10 dB, one pixel below the packing's floor at
:data:`FLOOR`, a zero (no-data) border.
"""

from __future__ import annotations

import datetime
import json

import numpy

from tests import orbit_fixture

NAME = "ICEYE_4EKGVZ_20251110T124428Z_7004635_X42_SLEDF"
HEIGHT, WIDTH = 256, 200
SHIP = (100, 120)
FLOOR = (50, 50)
#: the band's packing (amplitude = DN * SCALE + OFFSET) and calibration
SCALE, OFFSET, CF = 2.0, -10_000.0, 1e-6
SEA_DB, SHIP_DB = -20.0, 10.0
#: the ground grid: the top-left pixel centre, and metres per row / column
#: step -- rows run east-south-east (range), columns north-north-east
LAT0, LON0 = 1.30, 103.70
ROW_M, COL_M = 2.3, 2.5
_ROW_DIR, _COL_DIR = numpy.deg2rad(110.0), numpy.deg2rad(20.0)  # bearings
GCP_ROWS = [0.0, 85.0, 170.0, 255.0]
GCP_COLS = [0.0, 66.0, 133.0, 199.0]

#: the imaging geometry at :data:`TARGET` (the middle pixel), for the
#: synthetic orbit: right-looking down the rows (range)
INCIDENCE, SLANT_RANGE, LOOK = 24.99, 676_584.3, 110.0
CENTRE_TIME = datetime.datetime(2025, 11, 10, 12, 44, 42, 38000)


def latlon(row: float, col: float) -> tuple[float, float]:
    """The ground position of pixel centre ``(row, col)`` (pixel indices)."""
    north = row * ROW_M * numpy.cos(_ROW_DIR) + col * COL_M * numpy.cos(_COL_DIR)
    east = row * ROW_M * numpy.sin(_ROW_DIR) + col * COL_M * numpy.sin(_COL_DIR)
    lat = LAT0 + numpy.rad2deg(north / 6_371_008.8)
    lon = LON0 + numpy.rad2deg(east / (6_371_008.8 * numpy.cos(numpy.deg2rad(LAT0))))
    return float(lat), float(lon)


TARGET = latlon((HEIGHT - 1) / 2, (WIDTH - 1) / 2)


def _stamp(seconds: float) -> str:
    return (CENTRE_TIME + datetime.timedelta(seconds=seconds)).isoformat(timespec="microseconds") + "Z"


def _orbit_states() -> list[dict]:
    times, positions, velocities = orbit_fixture.states(
        *TARGET, look_bearing=LOOK, incidence=INCIDENCE, slant_range=SLANT_RANGE, right_looking=True
    )
    return [
        {"time": _stamp(t), "position": p.tolist(), "velocity": v.tolist()}
        for t, p, v in zip(times, positions, velocities)
    ]


PROPERTIES = {
    "datetime": "2025-11-10T12:44:42.038Z",
    "platform": "ICEYE-X42",
    "sar:instrument_mode": "spotlight",
    "sar:polarizations": ["VV"],
    "sar:observation_direction": "right",
    "sat:orbit_state": "descending",
    "sar:resolution_range": 0.59,
    "sar:resolution_azimuth": 0.5,
    "sar:looks_range": 1,
    "sar:looks_azimuth": 10,
    "view:off_nadir": 22.64,
    "view:incidence_angle": 24.99,
    "iceye:squint_angle": -0.045,
    "iceye:range": 676584.3,
    "iceye:scene_id": "4EKGVZ",
    "iceye:image_id": 7004635,
    "iceye:processing_mode": "SLEDF",
    "iceye:calibration_factor": CF,
    "iceye:zero_doppler_start_datetime": _stamp(-0.4),
    "iceye:zero_doppler_end_datetime": _stamp(0.4),
    "iceye:coordinate_frame": "ecef",
    "iceye:orbit_states": _orbit_states(),
}


def dn(db: float) -> int:
    """The pixel value whose sigma0 is ``db`` dB."""
    return round(((10 ** (db / 10) / CF) ** 0.5 - OFFSET) / SCALE)


def image() -> numpy.ndarray:
    data = numpy.full((HEIGHT, WIDTH), dn(SEA_DB), dtype="uint16")
    data[SHIP[0] - 3: SHIP[0] + 4, SHIP[1] - 1: SHIP[1] + 2] = dn(SHIP_DB)
    data[FLOOR] = 4000  # amplitude 4000 * 2 - 10000 < 0
    data[:8], data[-8:], data[:, :8], data[:, -8:] = 0, 0, 0, 0
    return data


def gcps(skip: tuple[tuple[float, float], ...] = ()):
    from rasterio.control import GroundControlPoint

    return [
        GroundControlPoint(row=r, col=c, x=latlon(r, c)[1], y=latlon(r, c)[0], z=63.0)
        for r in GCP_ROWS for c in GCP_COLS if (r, c) not in skip
    ]


def build(parent, *, skip_gcps: tuple[tuple[float, float], ...] = ()):
    """Write the product folder under ``parent``; returns its path. GCPs at
    ``skip_gcps`` ``(row, col)`` are left out, as a few real GRDs lack some."""
    import rasterio
    from rasterio.enums import Resampling

    folder = parent / NAME
    folder.mkdir()
    with rasterio.open(
        folder / f"{NAME}_GRD.tif", "w", driver="GTiff", width=WIDTH, height=HEIGHT, count=1,
        dtype="uint16", gcps=gcps(skip_gcps), crs="EPSG:4326", nodata=0,
        tiled=True, blockxsize=128, blockysize=128,
    ) as dst:
        dst.write(image(), 1)
        dst.scales, dst.offsets = (SCALE,), (OFFSET,)
        dst.update_tags(ICEYE_PROPERTIES=json.dumps(PROPERTIES), PRODUCT_TYPE="GRD-COG")
        dst.build_overviews([2, 4], Resampling.average)
    return folder
