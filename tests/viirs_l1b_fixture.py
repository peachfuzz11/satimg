"""A NASA VIIRS L1B DNB granule pair (``VJ102DNB`` + ``VJ103DNB``), built from
the real, minified NOAA SDR test granule and laid out per the LAADS file
specifications (VNP02DNB / VNP03DNB): ``observation_data/DNB_observations``
float32 W / (cm^2 sr), fill -999.9; ``geolocation_data`` latitude / longitude
float32, angles int16 x 0.01 with ``_FillValue``, per-pixel moon illumination
in hundredths of a percent; global ``time_coverage_*``, ``DayNightFlag``,
``startDirection``, ``GRingPoint*``. Same pixels, same geolocation -- so the
L1B reader can be checked against the SDR one.
"""

from __future__ import annotations

import numpy

L1B = "VJ102DNB.A2026305.1630.021.2026305232544.nc"
GEO = "VJ103DNB.A2026305.1630.021.2026305223722.nc"

_ANGLES = {
    "sensor_zenith": "SatelliteZenithAngle",
    "sensor_azimuth": "SatelliteAzimuthAngle",
    "solar_zenith": "SolarZenithAngle",
    "solar_azimuth": "SolarAzimuthAngle",
    "lunar_zenith": "LunarZenithAngle",
    "lunar_azimuth": "LunarAzimuthAngle",
}


def build(sdr_path, geo_path, out_dir) -> None:
    import h5py

    with h5py.File(sdr_path) as sdr, h5py.File(geo_path) as geo:
        radiance = sdr["All_Data/VIIRS-DNB-SDR_All/Radiance"][()]
        gran = sdr["Data_Products/VIIRS-DNB-SDR/VIIRS-DNB-SDR_Gran_0"].attrs
        g = geo["All_Data/VIIRS-DNB-GEO_All"]
        lat, lon = g["Latitude"][()], g["Longitude"][()]
        angles = {name: g[sdr_name][()] for name, sdr_name in _ANGLES.items()}
        moon = float(g["MoonIllumFraction"][()].ravel()[0])
        ring_lat, ring_lon = gran["G-Ring_Latitude"].ravel()[:4], gran["G-Ring_Longitude"].ravel()[:4]

    with h5py.File(out_dir / L1B, "w") as f:
        f.attrs["time_coverage_start"] = "2026-11-01T16:30:06.000Z"
        f.attrs["time_coverage_end"] = "2026-11-01T16:31:30.000Z"
        f.attrs["DayNightFlag"] = "Night"
        f.attrs["startDirection"] = "Descending"
        f.attrs["platform"] = "JPSS-1"
        f.attrs["orbit_number"] = numpy.int32(46398)
        f.attrs["GRingPointLatitude"] = ring_lat.astype("float32")
        f.attrs["GRingPointLongitude"] = ring_lon.astype("float32")
        data = numpy.where(radiance <= -999, numpy.float32(-999.9), radiance).astype("float32")
        obs = f.create_dataset("observation_data/DNB_observations", data=data, chunks=(16, 256))
        obs.attrs["_FillValue"] = numpy.float32(-999.9)
        obs.attrs["units"] = "Watts/cm^2/steradian"

    with h5py.File(out_dir / GEO, "w") as f:
        f.attrs["time_coverage_start"] = "2026-11-01T16:30:06.000Z"
        group = f.create_group("geolocation_data")
        for name, values in (("latitude", lat), ("longitude", lon)):
            ds = group.create_dataset(name, data=values.astype("float32"))
            ds.attrs["_FillValue"] = numpy.float32(-999.9)
        for name, values in {**angles, "moon_illumination_fraction": numpy.full(lat.shape, moon)}.items():
            scaled = numpy.round(numpy.asarray(values, dtype=float) / 0.01).astype("int16")
            ds = group.create_dataset(name, data=scaled)
            ds.attrs["_FillValue"] = numpy.int16(-32768)
            ds.attrs["scale_factor"] = numpy.float32(0.01)
            ds.attrs["add_offset"] = numpy.float32(0.0)
