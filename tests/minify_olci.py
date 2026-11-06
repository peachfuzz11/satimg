"""Cut a real Sentinel-3 OLCI EFR product down to a test fixture: a window of
:data:`ROWS` rows x :data:`COLS` columns starting at a tie column, with all 21
radiance bands, the tie-point geolocation and geometries cut to match, the
real per-pixel geolocation (to test against), and a minimal manifest whose
footprint is the window's. Then inject a bright 2 x 2 target at
:data:`TARGET` in the true-colour bands and a fill value at :data:`FILL`.

    uv run python -m tests.minify_olci SRC.SEN3 OUT_DIR ROW0 COL0

The committed fixture is S3A, 2024-09-16 10:10 UTC, rows 1920+, columns
4480+ (the Gulf of Bothnia coast near the swath edge), from the public
Sentinel-3 archive on AWS (MEEO).
"""

import os
import sys

import numpy

ROWS, COLS = 128, 257  # 257: the window ends on a tie column (every 64th)
TARGET = (60, 100)
TARGET_DN = 30000  # bright enough to saturate the visual in every true colour band
FILL = (5, 5)


def _copy_attrs(src, dst):
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def minify(src_dir, out_dir, row0, col0):
    import h5py

    name = os.path.basename(os.path.normpath(src_dir))
    out = os.path.join(out_dir, name)
    os.makedirs(out, exist_ok=True)
    with h5py.File(os.path.join(src_dir, "tie_geo_coordinates.nc")) as f:
        step_ac = int(numpy.ravel(f.attrs["ac_subsampling_factor"])[0])
    assert col0 % step_ac == 0, "start on a tie column"
    rows = slice(row0, row0 + ROWS)
    cols = slice(col0, col0 + COLS)
    tie_cols = slice(col0 // step_ac, (col0 + COLS - 1) // step_ac + 1)

    for band in range(1, 22):
        var = f"Oa{band:02d}_radiance"
        with h5py.File(os.path.join(src_dir, f"{var}.nc")) as src, h5py.File(os.path.join(out, f"{var}.nc"), "w") as dst:
            _copy_attrs(src, dst)
            data = src[var][rows, cols]
            if band in (4, 6, 8):
                data[TARGET[0]:TARGET[0] + 2, TARGET[1]:TARGET[1] + 2] = TARGET_DN
            data[FILL] = 65535
            ds = dst.create_dataset(var, data=data, chunks=(64, 128), compression="gzip", shuffle=True)
            _copy_attrs(src[var], ds)
    for fname, cut in (("tie_geo_coordinates.nc", tie_cols), ("tie_geometries.nc", tie_cols), ("geo_coordinates.nc", cols)):
        with h5py.File(os.path.join(src_dir, fname)) as src, h5py.File(os.path.join(out, fname), "w") as dst:
            _copy_attrs(src, dst)
            for var, ds in src.items():
                if ds.ndim != 2:
                    continue
                part = dst.create_dataset(var, data=ds[rows, cut], compression="gzip", shuffle=True)
                _copy_attrs(ds, part)
    with h5py.File(os.path.join(out, "geo_coordinates.nc")) as f:
        lat, lon = f["latitude"][()] * 1e-6, f["longitude"][()] * 1e-6
    corners = [(0, 0), (0, COLS - 1), (ROWS - 1, COLS - 1), (ROWS - 1, 0), (0, 0)]
    pos = " ".join(f"{lat[r, c]:.5f} {lon[r, c]:.5f}" for r, c in corners)
    with open(os.path.join(out, "xfdumanifest.xml"), "w") as f:
        f.write(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<xfdu:XFDU xmlns:xfdu="urn:ccsds:schema:xfdu:1" xmlns:gml="http://www.opengis.net/gml" '
            'xmlns:sentinel-safe="http://www.esa.int/safe/sentinel/1.1"><metadataSection>'
            '<metadataObject ID="measurementFrameSet"><metadataWrap><xmlData><sentinel-safe:frameSet>'
            f"<sentinel-safe:footPrint><gml:posList>{pos}</gml:posList></sentinel-safe:footPrint>"
            "</sentinel-safe:frameSet></xmlData></metadataWrap></metadataObject></metadataSection></xfdu:XFDU>\n"
        )


if __name__ == "__main__":
    minify(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
