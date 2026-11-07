"""A small synthetic NISAR L2 GCOV ``.h5``, laid out per the GCOV product
specification (JPL D-102274) -- real granules are several GB and need an
Earthdata login, so the tests build their own.

A descending, left-looking frame off Lofoten on a UTM 33N grid, 20 m pixels,
dual-pol (HHHH / HVHV): sea everywhere, one bright 'ship' at :data:`SHIP`, and
a ``NaN`` (no-data) strip down the left edge.
"""

from __future__ import annotations

import numpy

NAME = "NISAR_L2_PR_GCOV_008_029_D_010_4005_DHDH_A_20260801T061010_20260801T061025_P05023_F_F_J_001.h5"

HEIGHT, WIDTH = 400, 300
SPACING = 20.0
#: centre of the top-left pixel (CF convention)
X0, Y0 = 450_010.0, 7_600_010.0
EPSG = 32633
#: (row, col) of the bright target
SHIP = (200, 150)
SEA_POWER = 10 ** (-25 / 10)
SHIP_POWER = 10 ** (10 / 10)

PLATFORM_HEADING = 192.0  # descending
LOOK = (PLATFORM_HEADING - 90.0) % 360  # left-looking
#: incidence (degrees) and slant range (m) at sea level; the cube's layers above
#: and below differ, so a test can tell the sea-level interpolation happened
INCIDENCE_0 = 35.0
SLANT_RANGE_0 = 900_000.0
CUBE_HEIGHTS = (-500.0, 500.0, 1500.0)
PLATFORM_SPEED = 7_600.0


def _s(text: str):
    return numpy.bytes_(text)


def build(path) -> None:
    import h5py

    xs = X0 + SPACING * numpy.arange(WIDTH)
    ys = Y0 - SPACING * numpy.arange(HEIGHT)
    hh = numpy.full((HEIGHT, WIDTH), SEA_POWER, dtype="float32")
    hh[SHIP[0] - 2: SHIP[0] + 3, SHIP[1] - 1: SHIP[1] + 2] = SHIP_POWER
    hh[:, :10] = numpy.nan
    hv = hh / 10

    with h5py.File(path, "w") as f:
        ident = f.create_group("/science/LSAR/identification")
        for key, value in {
            "zeroDopplerStartTime": "2026-08-01T06:10:10.000000000",
            "zeroDopplerEndTime": "2026-08-01T06:10:25.500000000",
            "granuleId": NAME[:-3],
            "lookDirection": "Left",
            "orbitPassDirection": "Descending",
            "boundingPolygon": "POLYGON Z ((14.8 68.55 0, 14.95 68.55 0, 14.95 68.47 0, 14.8 68.47 0, 14.8 68.55 0))",
        }.items():
            ident[key] = _s(value)
        ident["absoluteOrbitNumber"] = numpy.uint32(5023)
        ident["trackNumber"] = numpy.uint8(29)
        ident["frameNumber"] = numpy.uint16(10)

        grid = f.create_group("/science/LSAR/GCOV/grids/frequencyA")
        grid["xCoordinates"] = xs
        grid["yCoordinates"] = ys
        grid["xCoordinateSpacing"] = SPACING
        grid["yCoordinateSpacing"] = -SPACING
        grid["listOfPolarizations"] = numpy.array([b"HH", b"HV"])
        grid["listOfCovarianceTerms"] = numpy.array([b"HHHH", b"HVHV"])
        grid["projection"] = numpy.uint32(EPSG)
        grid["projection"].attrs["epsg_code"] = EPSG
        for term, data in (("HHHH", hh), ("HVHV", hv)):
            grid.create_dataset(term, data=data, chunks=(128, 128), compression="gzip")
        grid.create_dataset("mask", data=numpy.ones((HEIGHT, WIDTH), "uint8"), chunks=(128, 128))

        radar = f.create_group("/science/LSAR/GCOV/metadata/radarGrid")
        cxs = numpy.arange(X0 - 2000, X0 + SPACING * WIDTH + 2000, 1000.0)
        cys = numpy.arange(Y0 + 2000, Y0 - SPACING * HEIGHT - 2000, -1000.0)
        heights = numpy.array(CUBE_HEIGHTS)
        radar["xCoordinates"] = cxs
        radar["yCoordinates"] = cys
        radar["heightAboveEllipsoid"] = heights
        shape = (len(heights), len(cys), len(cxs))
        # values move linearly with height, so only an interpolation to sea
        # level -- halfway between the -500 m and 500 m layers -- gives *_0
        per_metre = heights[:, None, None] * numpy.ones(shape)
        inc = INCIDENCE_0 - 0.001 * per_metre
        radar["incidenceAngle"] = inc.astype("float32")
        radar["elevationAngle"] = (inc - 3).astype("float32")
        radar["slantRange"] = SLANT_RANGE_0 - 0.5 * per_metre
        sin_inc = numpy.sin(numpy.deg2rad(inc))
        # LOS points target -> sensor, i.e. against the look direction
        radar["losUnitVectorX"] = (-numpy.sin(numpy.deg2rad(LOOK)) * sin_inc).astype("float32")
        radar["losUnitVectorY"] = (-numpy.cos(numpy.deg2rad(LOOK)) * sin_inc).astype("float32")
        radar["alongTrackUnitVectorX"] = numpy.full(shape, numpy.sin(numpy.deg2rad(PLATFORM_HEADING)), "float32")
        radar["alongTrackUnitVectorY"] = numpy.full(shape, numpy.cos(numpy.deg2rad(PLATFORM_HEADING)), "float32")
        radar["groundTrackVelocity"] = numpy.full(shape[1:], 6_900.0)

        orbit = f.create_group("/science/LSAR/GCOV/metadata/orbit")
        times = numpy.arange(0.0, 60.0, 10.0)
        orbit["time"] = times
        orbit["time"].attrs["units"] = _s("seconds since 2026-08-01T06:09:50")
        heading = numpy.deg2rad(PLATFORM_HEADING)
        orbit["velocity"] = numpy.tile(
            [PLATFORM_SPEED * numpy.sin(heading), PLATFORM_SPEED * numpy.cos(heading), 0.0], (len(times), 1)
        )
        orbit["position"] = numpy.zeros((len(times), 3))
