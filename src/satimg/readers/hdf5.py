"""HDF5 / netCDF4 access through ``h5py`` -- by path on disk, or through a
zip archive's file objects."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

import dask.array
import h5py
import numpy
import xarray

if TYPE_CHECKING:  # pragma: no cover
    from satimg.source.base import Source


@contextmanager
def open_h5(source: Source, relpath: str) -> Iterator[h5py.File]:
    """``relpath`` opened read-only with ``h5py`` -- by path where the file is
    on disk, else (zip-native) through the archive's file object."""
    if source.is_directory:
        with h5py.File(source.raster_path(relpath), "r") as f:
            yield f
    else:
        with source.open(relpath) as raw, h5py.File(raw, "r") as f:
            yield f


class LazyH5Dataset:
    """An HDF5 dataset dask can slice, opening its file on first read and
    again after :meth:`close` -- so a lazy view outlives the loop that made
    it, as GDAL-backed ones do (they reopen by path). Reads hold a lock:
    HDF5 is not thread-safe."""

    def __init__(self, source: Source, relpath: str, dataset: str):
        self._source, self._relpath, self._dataset = source, relpath, dataset
        self._lock = threading.Lock()
        self._handle = self._file = None
        with self._open() as f:
            ds = f[dataset]
            self.shape, self.dtype, self.ndim = ds.shape, ds.dtype, ds.ndim
            self.chunks = ds.chunks

    def _open(self):
        self._handle = (
            open(self._source.raster_path(self._relpath), "rb")
            if self._source.is_directory
            else self._source.open_detached(self._relpath)
        )
        self._file = h5py.File(self._handle, "r")
        return self

    def __enter__(self):
        return self._file

    def __exit__(self, *exc):
        self.close()

    def __getitem__(self, key):
        with self._lock:
            if self._file is None:
                self._open()
            return self._file[self._dataset][key]

    def close(self):
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._handle.close()
            self._file = self._handle = None


def open_h5_array(source: Source, relpath: str, dataset: str, tile: int | tuple[int, int]) -> xarray.DataArray:
    """HDF5 / netCDF4 ``dataset`` of ``relpath`` as a lazy ``(band, y, x)``
    (or ``(y, x)``) array read by ``h5py`` through dask, chunked to ``tile``
    with every band in one chunk -- for datasets GDAL's HDF5 driver cannot
    name (groups with spaces, as in HDF-EOS5 swaths). Works zip-native too.
    ``close()`` closes the file; a later read reopens it.
    """
    tw, th = (tile, tile) if isinstance(tile, int) else tile
    lazy = LazyH5Dataset(source, relpath, dataset)
    chunks = (th, tw) if lazy.ndim == 2 else (-1, th, tw)
    arr = dask.array.from_array(lazy, chunks=chunks, lock=False, asarray=True)
    da = xarray.DataArray(arr, dims=("y", "x") if lazy.ndim == 2 else ("band", "y", "x"))
    da.set_close(lazy.close)
    return da


def read_scaled(dataset, fill_below: float | None = None, key=()) -> numpy.ndarray:
    """An HDF5 / netCDF4 variable -- or just its ``key`` selection (rows,
    slices: anything ``h5py`` indexes by), so a full-resolution variable
    need not be read whole -- as ``float``, its CF ``_FillValue`` (and
    anything at or below ``fill_below``) as ``NaN``, ``scale_factor`` /
    ``add_offset`` applied -- ``h5py`` applies none of them itself."""
    raw = dataset[key]
    data = raw.astype(float)
    attrs = dataset.attrs
    if "_FillValue" in attrs:
        data[raw == numpy.asarray(attrs["_FillValue"]).ravel()[0]] = numpy.nan
    if fill_below is not None:
        data[data <= fill_below] = numpy.nan
    if "scale_factor" in attrs:
        data *= float(numpy.asarray(attrs["scale_factor"]).ravel()[0])
    if "add_offset" in attrs:
        data += float(numpy.asarray(attrs["add_offset"]).ravel()[0])
    return data


def find_dataset(h5, name: str) -> str:
    """The full path of the first dataset called ``name`` anywhere in
    ``h5`` -- for products whose group layout varies (MODIS C7's
    HDF-EOS5 swath groups)."""
    found = []

    def visit(path, obj):
        if isinstance(obj, h5py.Dataset) and path.rsplit("/", 1)[-1] == name:
            found.append(path)
            return True
        return None

    h5.visititems(visit)
    if not found:
        raise KeyError(f"no dataset {name!r} in {h5.filename}")
    return "/" + found[0]
