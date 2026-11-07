"""IO helpers shared by the product readers, one module per kind of file:

- :mod:`~satimg.readers.raster` -- GDAL rasters, lazily, band-stacked
- :mod:`~satimg.readers.hdf5` -- HDF5 / netCDF4 through ``h5py``
- :mod:`~satimg.readers.vrt` -- VRT wrappers that georeference a raster
- :mod:`~satimg.readers.thumbnail` -- uniformly sized thumbnails
"""

from satimg.readers.hdf5 import LazyH5Dataset, find_dataset, open_h5, open_h5_array, read_scaled
from satimg.readers.raster import CHUNK_PX, keep_open, merge_bands, open_band
from satimg.readers.thumbnail import THUMBNAIL_SIZE, fit_thumbnail, load_thumbnail
from satimg.readers.vrt import vrt

__all__ = [
    "CHUNK_PX", "keep_open", "merge_bands", "open_band",
    "LazyH5Dataset", "find_dataset", "open_h5", "open_h5_array", "read_scaled",
    "THUMBNAIL_SIZE", "fit_thumbnail", "load_thumbnail",
    "vrt",
]
