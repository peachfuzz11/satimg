"""Cut a real VIIRS DNB granule (SVDNB + GDNBO) down to a test fixture: the
first :data:`ROWS` rows (whole scans) and :data:`COLS` columns -- real
attributes, geolocation, angles and radiance, so the swath-edge noise columns
and scan structure are the real thing -- then inject known boat lights, a
particle hit and a fill value at :data:`BOATS` / :data:`PARTICLE` /
:data:`FILL`.

    uv run python -m tests.minify_viirs SVDNB_....h5 GDNBO_....h5 OUT_DIR
"""

import os
import sys

import numpy

ROWS, COLS = 64, 256
#: (row, col) of each injected boat light: a 3 x 3 blurred spot, nW / (cm^2 sr)
BOATS = [(20, 100), (40, 180), (52, 60)]
BOAT_PEAK = 60.0
#: (row, col) of an injected particle hit: one pixel, nothing around it
PARTICLE = (30, 140)
#: (row, col) of an injected fill value
FILL = (10, 200)

_GEO_KEEP = (
    "Latitude", "Longitude", "SatelliteZenithAngle", "SatelliteAzimuthAngle", "SolarZenithAngle",
    "SolarAzimuthAngle", "LunarZenithAngle", "LunarAzimuthAngle", "MoonIllumFraction", "MoonPhaseAngle",
)


def _copy_attrs(src, dst):
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def minify(sdr_path, geo_path, out_dir):
    import h5py

    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(sdr_path) as src, h5py.File(os.path.join(out_dir, os.path.basename(sdr_path)), "w") as dst:
        _copy_attrs(src, dst)
        gran = "Data_Products/VIIRS-DNB-SDR/VIIRS-DNB-SDR_Gran_0"
        dst.create_dataset(gran, data=numpy.zeros(1, "uint8"))
        _copy_attrs(src[gran], dst[gran])
        radiance = src["All_Data/VIIRS-DNB-SDR_All/Radiance"][:ROWS, :COLS].astype("float32")
        spot = numpy.array([[0.1, 0.3, 0.1], [0.3, 1.0, 0.3], [0.1, 0.3, 0.1]], "float32") * BOAT_PEAK * 1e-9
        for r, c in BOATS:
            radiance[r - 1:r + 2, c - 1:c + 2] += spot
        radiance[PARTICLE] += BOAT_PEAK * 1e-9
        radiance[FILL] = -999.3
        dst.create_dataset("All_Data/VIIRS-DNB-SDR_All/Radiance", data=radiance, compression="gzip", shuffle=True)
    with h5py.File(geo_path) as src, h5py.File(os.path.join(out_dir, os.path.basename(geo_path)), "w") as dst:
        _copy_attrs(src, dst)
        for name in _GEO_KEEP:
            data = src[f"All_Data/VIIRS-DNB-GEO_All/{name}"]
            data = data[:ROWS, :COLS] if data.ndim == 2 else data[()]
            dst.create_dataset(f"All_Data/VIIRS-DNB-GEO_All/{name}", data=data, compression="gzip", shuffle=True)


if __name__ == "__main__":
    minify(*sys.argv[1:4])
