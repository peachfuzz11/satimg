"""A small synthetic MODIS Collection 7 250 m L1B granule (``MOD02QKM`` /
``MYD02QKM``), laid out per the MCST Level 1B Product User's Guide (C7):
HDF-EOS5 swath groups (names with spaces), ``EV_250_RefSB`` uint16
``(2, 40 * scans, 4 * frames)`` with ``reflectance_scales`` /
``reflectance_offsets`` / ``radiance_*``, 65535 fill, and 1 km ``Latitude`` /
``Longitude`` with the 1.5-pixel track fractional offset. Real granules are
~280 MB and need an Earthdata login.

The geometry is a made-up swath over the southern North Sea with MODIS's
shape: 10 detector rows per scan, each scan wider along track towards the
swath edges, so neighbouring scans overlap there (the bow-tie).
"""

from __future__ import annotations

import numpy

TERRA = "MOD02QKM.A2026305.1035.007.2026305170237.nc"
AQUA = "MYD02QKM.A2026305.1250.007.2026306151959.nc"

SCANS, FRAMES = 3, 64  # 1 km: 30 x 64; 250 m: 120 x 256
ROWS, COLS = 40 * SCANS, 4 * FRAMES
REFLECTANCE_SCALES = (5.3e-5, 3.2e-5)
REFLECTANCE_OFFSETS = (0.0, 0.0)
WATER_DN = 700  # NIR reflectance ~2%
SHIP = (60, 130)  # 250 m (row, col): a 2 x 2 bright target
SHIP_DN = 12_000  # NIR reflectance ~38%
FILL_ROWS = slice(0, 2)  # 65535 across the top two rows


def geolocation() -> tuple[numpy.ndarray, numpy.ndarray]:
    """1 km latitude / longitude, ``(10 * SCANS, FRAMES)``."""
    rows = numpy.arange(10 * SCANS)
    scan, detector = rows // 10, rows % 10
    u = (numpy.arange(FRAMES) - (FRAMES - 1) / 2) / (FRAMES / 2)  # -1 .. 1 across the swath
    growth = 1 + 1.0 * u ** 2  # along-track footprint widens to 2x at the edge
    lat = 53.0 - 0.09 * scan[:, None] - (detector[:, None] - 4.5) * 0.009 * growth[None, :]
    lon = 2.5 + 0.03 * (numpy.arange(FRAMES) - FRAMES / 2) * (1 + 0.3 * u ** 2)[None, :]
    return lat, numpy.broadcast_to(lon, lat.shape).copy()


def build(path) -> None:
    import h5py

    data = numpy.full((2, ROWS, COLS), WATER_DN, dtype="uint16")
    data[0] = WATER_DN * 2  # red over water: brighter than NIR
    data[1, SHIP[0]:SHIP[0] + 2, SHIP[1]:SHIP[1] + 2] = SHIP_DN
    data[:, FILL_ROWS, :] = 65535
    lat, lon = geolocation()
    swath = "/HDFEOS/SWATHS/MODIS_SWATH_Type_L1B"
    with h5py.File(path, "w") as f:
        f.attrs["HDFEOS_FractionalOffset_10*nscans_MODIS_SWATH_Type_L1B"] = numpy.float32(1.5)
        f.attrs["HDFEOS_FractionalOffset_Max_EV_frames_MODIS_SWATH_Type_L1B"] = numpy.float32(0.0)
        f.attrs["DAYNIGHTFLAG"] = "Day"
        ev = f.create_dataset(f"{swath}/Data Fields/EV_250_RefSB", data=data, chunks=(2, 40, 128), compression="gzip")
        ev.attrs["_FillValue"] = numpy.uint16(65535)
        ev.attrs["valid_range"] = numpy.array([0, 32767], dtype="uint16")
        ev.attrs["band_names"] = "1,2"
        ev.attrs["reflectance_scales"] = numpy.array(REFLECTANCE_SCALES, dtype="float32")
        ev.attrs["reflectance_offsets"] = numpy.array(REFLECTANCE_OFFSETS, dtype="float32")
        ev.attrs["radiance_scales"] = numpy.array([0.026, 0.0099], dtype="float32")
        ev.attrs["radiance_offsets"] = numpy.array([0.0, 0.0], dtype="float32")
        for name, values in (("Latitude", lat), ("Longitude", lon)):
            ds = f.create_dataset(f"{swath}/Geolocation Fields/{name}", data=values.astype("float32"))
            ds.attrs["_FillValue"] = numpy.float32(-999.0)


TERRA_GEO = "MOD03.A2026305.1035.007.2026305165012.nc"
SOLAR_ZENITH = 60.0


def build_geolocation(path) -> None:
    """The matching ``M?D03``: the same 1 km latitude / longitude, plus sun
    and view angles as int16 x 0.01 (MOD03's encoding)."""
    import h5py

    lat, lon = geolocation()
    u = numpy.broadcast_to((numpy.arange(FRAMES) - (FRAMES - 1) / 2) / (FRAMES / 2), lat.shape)
    angles = {
        "SolarZenith": numpy.full(lat.shape, SOLAR_ZENITH),
        "SolarAzimuth": numpy.full(lat.shape, 160.0),
        "SensorZenith": 65.0 * numpy.abs(u),
        "SensorAzimuth": numpy.where(u < 0, -100.0, 80.0),
    }
    swath = "/HDFEOS/SWATHS/MODIS_Swath_Type_GEO"
    with h5py.File(path, "w") as f:
        for name, values in (("Latitude", lat), ("Longitude", lon)):
            ds = f.create_dataset(f"{swath}/Geolocation Fields/{name}", data=values.astype("float32"))
            ds.attrs["_FillValue"] = numpy.float32(-999.0)
        for name, values in angles.items():
            ds = f.create_dataset(f"{swath}/Data Fields/{name}", data=numpy.round(values / 0.01).astype("int16"))
            ds.attrs["_FillValue"] = numpy.int16(-32767)
            ds.attrs["scale_factor"] = numpy.float32(0.01)
