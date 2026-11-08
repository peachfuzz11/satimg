from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy

BASE_DIR = Path(__file__).resolve().parent


@contextmanager
def h5_reads():
    """Record every ``h5py`` dataset read inside the block, as ``(dataset
    name, elements read, elements in the dataset)`` -- to check a product
    opens without reading its full-resolution arrays whole."""
    reads = []
    original = h5py.Dataset.__getitem__

    def spy(self, key, *args, **kwargs):
        out = original(self, key, *args, **kwargs)
        reads.append((self.name, numpy.size(out), self.size))
        return out

    h5py.Dataset.__getitem__ = spy
    try:
        yield reads
    finally:
        h5py.Dataset.__getitem__ = original
