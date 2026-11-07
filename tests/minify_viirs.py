"""Cut a real VIIRS DNB L1B granule pair (``V??02DNB`` + ``V??03DNB``) down to
a test fixture: the first :data:`ROWS` rows (whole scans) and :data:`COLS`
columns -- real attributes, geolocation, angles and radiance, so the
swath-edge noise columns and scan structure are the real thing -- then inject
known boat lights, a particle hit and a fill value at :data:`BOATS` /
:data:`PARTICLE` / :data:`FILL`.

    uv run python -m tests.minify_viirs VJ102DNB....nc VJ103DNB....nc OUT_DIR

The committed fixture (``VJ102DNB.A2026305.1630.021...``) holds a real
NOAA-20 pass -- 2026-11-01 16:30 UTC over the Sea of Japan -- cut this way,
its radiance, geolocation and angles taken from NOAA's distribution of the
same granule (no Earthdata login was to hand) and written in the LAADS
VNP02DNB / VNP03DNB layout.
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

_RADIANCE = "observation_data/DNB_observations"
_GEO_KEEP = (
    "latitude", "longitude", "sensor_zenith", "sensor_azimuth", "solar_zenith", "solar_azimuth",
    "lunar_zenith", "lunar_azimuth", "moon_illumination_fraction", "moon_phase_angle",
)


def _copy_attrs(src, dst):
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def minify(l1b_path, geo_path, out_dir):
    import h5py

    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(l1b_path) as src, h5py.File(os.path.join(out_dir, os.path.basename(l1b_path)), "w") as dst:
        _copy_attrs(src, dst)
        radiance = src[_RADIANCE][:ROWS, :COLS].astype("float32")
        spot = numpy.array([[0.1, 0.3, 0.1], [0.3, 1.0, 0.3], [0.1, 0.3, 0.1]], "float32") * BOAT_PEAK * 1e-9
        for r, c in BOATS:
            radiance[r - 1:r + 2, c - 1:c + 2] += spot
        radiance[PARTICLE] += BOAT_PEAK * 1e-9
        radiance[FILL] = -999.9
        out = dst.create_dataset(_RADIANCE, data=radiance, compression="gzip", shuffle=True)
        _copy_attrs(src[_RADIANCE], out)
    with h5py.File(geo_path) as src, h5py.File(os.path.join(out_dir, os.path.basename(geo_path)), "w") as dst:
        _copy_attrs(src, dst)
        for name in _GEO_KEEP:
            path = f"geolocation_data/{name}"
            if path not in src:
                continue
            out = dst.create_dataset(path, data=src[path][:ROWS, :COLS], compression="gzip", shuffle=True)
            _copy_attrs(src[path], out)


if __name__ == "__main__":
    minify(*sys.argv[1:4])
