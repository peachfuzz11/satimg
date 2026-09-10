"""The GDAL rasters a product opens must be released -- on ``with`` exit
(``__exit__``) and by ``open_zip`` -- so a loop over many products does not
leak file descriptors.

fd counting is Linux-only (``/proc/self/fd``); ``psutil`` is not a dependency.
"""

import gc
import os
import sys
import zipfile
from pathlib import Path

import pytest

import satimg
from tests.conftest import PRODUCT_PATHS

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="fd counting is Linux-only"
)


def _open_fds() -> int:
    gc.collect()
    return len(os.listdir("/proc/self/fd"))


def _fresh(key: str):
    """A brand-new product instance -- never the module-scoped fixtures, whose
    sharing would let one test's ``_close()`` break another."""
    path = PRODUCT_PATHS[key]
    if not os.path.exists(path):
        pytest.skip(f"test product missing: {path}")
    return satimg.open(str(path))


ALL = ["sentinel1_iw", "sentinel2_l1c", "landsat"]


@pytest.mark.parametrize("key", ALL)
def test_close_shuts_the_rasters_and_clears_the_cache(key):
    base = _open_fds()
    p = _fresh(key)
    _ = p.raw
    _ = p.visual
    assert _open_fds() > base, "raw/visual should have opened rasters"

    p._close()

    assert _open_fds() - base <= 1, "every raster should be closed"
    assert "raw" not in p.__dict__ and "visual" not in p.__dict__
    p._close()  # idempotent


@pytest.mark.parametrize("key", ALL)
def test_views_rebuild_after_close(key):
    p = _fresh(key)
    shape = p.raw.shape
    p._close()
    assert p.raw.shape == shape  # rebuilds lazily from files still on disk


@pytest.mark.parametrize("key", ALL)
def test_context_manager_releases_fds(key):
    base = _open_fds()
    with _fresh(key) as p:
        _ = p.visual
        assert _open_fds() > base
    assert _open_fds() - base <= 1
    assert "visual" not in p.__dict__


@pytest.mark.parametrize("key", ALL)
def test_loop_over_products_does_not_leak(key):
    base = _open_fds()
    for _ in range(10):
        with _fresh(key) as p:
            _ = p.visual
    assert _open_fds() - base <= 1


@pytest.fixture(scope="module")
def landsat_zip(tmp_path_factory):
    src = Path(PRODUCT_PATHS["landsat"])
    if not src.exists():
        pytest.skip(f"test product missing: {src}")
    out = tmp_path_factory.mktemp("zips") / (src.name + ".zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in src.rglob("*"):
            if f.is_file():
                z.write(f, arcname=str(Path(src.name) / f.relative_to(src)))
    return str(out)


def test_open_zip_releases_handles_and_cleans_tempdir(landsat_zip, tmp_path):
    base = _open_fds()
    before = set(os.listdir(tmp_path))

    with satimg.open_zip(landsat_zip, dest=str(tmp_path)) as p:
        _ = p.visual
        assert os.listdir(tmp_path), "extraction dir should exist inside the block"

    gc.collect()
    assert set(os.listdir(tmp_path)) == before, "temp dir not cleaned up"
    assert _open_fds() - base <= 1, "file descriptors leaked past open_zip"


def test_open_zip_loop_does_not_leak(landsat_zip, tmp_path):
    base = _open_fds()
    for _ in range(8):
        with satimg.open_zip(landsat_zip, dest=str(tmp_path)) as p:
            _ = p.visual  # opens rasters lazily; open_zip closes them on exit
            next(iter(p.patches(128)))
    gc.collect()
    assert os.listdir(tmp_path) == [], "temp dirs left behind"
    assert _open_fds() - base <= 1
