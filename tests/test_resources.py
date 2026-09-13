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
    # 3 iterations is enough to catch a per-iteration leak trend without
    # paying for a long loop of real product opens.
    base = _open_fds()
    for _ in range(3):
        with _fresh(key) as p:
            _ = p.visual
    assert _open_fds() - base <= 1


def _build_zip(key: str, tmp_path_factory) -> str:
    src = Path(PRODUCT_PATHS[key])
    if not src.exists():
        pytest.skip(f"test product missing: {src}")
    out = tmp_path_factory.mktemp("zips") / (src.name + ".zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in src.rglob("*"):
            if f.is_file():
                z.write(f, arcname=str(Path(src.name) / f.relative_to(src)))
    return str(out)


@pytest.fixture(scope="module")
def landsat_zip(tmp_path_factory):
    return _build_zip("landsat", tmp_path_factory)


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
    # 3 iterations is enough to catch a per-iteration leak trend without
    # paying for a long loop of real zip extractions.
    base = _open_fds()
    for _ in range(3):
        with satimg.open_zip(landsat_zip, dest=str(tmp_path)) as p:
            _ = p.visual  # opens rasters lazily; open_zip closes them on exit
            next(iter(p.patches(128)))
    gc.collect()
    assert os.listdir(tmp_path) == [], "temp dirs left behind"
    assert _open_fds() - base <= 1


def test_open_detects_a_zip_and_cleans_up_like_open_zip(landsat_zip):
    # satimg.open() auto-detects a zip archive (by content, not extension)
    # and extracts it just like satimg.open_zip(path, extract=True) would.
    base = _open_fds()

    with satimg.open(landsat_zip) as p:
        temp_root = os.path.dirname(p.path)
        assert os.path.isdir(temp_root), "should have extracted to a temp dir"
        _ = p.visual

    gc.collect()
    assert not os.path.exists(temp_root), "temp dir not cleaned up"
    assert _open_fds() - base <= 1, "file descriptors leaked past open()"


def test_open_zip_extract_false_never_writes_to_disk_and_releases_handles(
    landsat_zip, tmp_path
):
    base = _open_fds()
    before = set(os.listdir(tmp_path))

    with satimg.open_zip(landsat_zip, dest=str(tmp_path), extract=False) as p:
        _ = p.timestamp, p.footprint, p.transformer, p.thumbnail()
        assert set(os.listdir(tmp_path)) == before, "extract=False must never write to dest"

    gc.collect()
    assert _open_fds() - base <= 1, "file descriptors leaked past open_zip(extract=False)"


def test_open_zip_extract_false_loop_does_not_leak(landsat_zip):
    base = _open_fds()
    for _ in range(3):
        with satimg.open_zip(landsat_zip, extract=False) as p:
            _ = p.timestamp, p.footprint, p.transformer, p.thumbnail()
    gc.collect()
    assert _open_fds() - base <= 1


@pytest.mark.parametrize("key", ALL)
def test_patches_closes_its_own_raster_when_the_loop_is_exhausted(key):
    # patches()/patches_at() open a raw/visual view of their own, separate
    # from product.raw/.visual, and must close it themselves when the loop
    # ends -- with no product._close() / `with product:` involved at all.
    p = _fresh(key)
    base = _open_fds()

    for patch in p.patches_at([(64, 64)], 64):
        _ = patch.raw.values
        _ = patch.visual.values
        assert _open_fds() > base, "the loop's own rasters should be open"

    assert _open_fds() - base <= 1, "exhausting the loop should close them"
    assert "raw" not in p.__dict__ and "visual" not in p.__dict__, (
        "patches_at must never touch product.raw / product.visual"
    )


@pytest.mark.parametrize("key", ALL)
def test_patches_closes_its_own_raster_on_early_close(key):
    # breaking out of (or otherwise abandoning) a patches() loop must still
    # release its rasters -- generator.close() is what a `for ... break`
    # triggers once the generator is garbage-collected; call it directly here
    # so the test isn't timing-dependent on when that GC happens.
    p = _fresh(key)
    base = _open_fds()

    gen = p.patches(64)
    patch = next(gen)
    _ = patch.raw.values
    _ = patch.visual.values
    assert _open_fds() > base

    gen.close()
    assert _open_fds() - base <= 1, "closing the generator early should release it"
