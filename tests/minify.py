import os
import shutil

import numpy
import rasterio

from tests.test_helper import BASE_DIR

if __name__ == "__main__":

    filename = "data/products_minified/LC09_L1TP_194022_20251013_20251013_02_T1"
    folder = os.path.join(BASE_DIR, "data", filename)
    mini_folder = os.path.join(BASE_DIR, "data", "products_minified", filename)
    os.makedirs(mini_folder, exist_ok=True)
    for item in os.listdir(folder):
        f = os.path.join(folder, item)
        t = os.path.join(mini_folder, item)
        if item.endswith(".TIF"):
            with rasterio.open(f) as src:
                meta = src.meta.copy()
                zero_data = numpy.zeros((src.count, src.height, src.width), dtype=src.dtypes[0])
            meta.update(compress='deflate')
            with rasterio.open(t, "w", **meta) as dst:
                dst.write(zero_data)
        else:
            shutil.copy(f, t)
