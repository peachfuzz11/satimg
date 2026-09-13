"""``merge_bands`` must open each band pre-chunked to the caller's ``tile``,
never collapsed to one whole-band chunk -- and ``Product.patches()`` /
``patches_at()`` must open their own raw / visual view chunked to exactly the
loop's own window size (an argument, not a size read back off the product).

A single chunk defeats every windowed read: rechunking a *single* dask chunk
into tiles still depends on one task that reads the entire band, so every
``patch.raw`` / ``patch.visual`` access would re-decode the whole scene
instead of just its own tile. See ``satimg.readers.merge_bands`` and
``satimg.product.Product.patches`` / ``patches_at``.
"""

import os

import pytest

import satimg
from satimg.readers import CHUNK_PX
from tests.conftest import PRODUCT_PATHS

PRODUCTS = ["sentinel1_iw", "sentinel2_l1c", "landsat"]


def _fresh(key: str):
    """A brand-new product instance -- never the module-scoped fixtures, so one
    test's forced raw/visual build can't leak into another's laziness check."""
    path = PRODUCT_PATHS[key]
    if not os.path.exists(path):
        pytest.skip(f"test product missing: {path}")
    return satimg.open(str(path))


@pytest.mark.parametrize("name", PRODUCTS)
def test_raw_bands_are_tile_chunked_not_one_chunk(request, name):
    # product.raw / product.visual (standalone, outside any loop) open at the
    # sane CHUNK_PX default.
    product = request.getfixturevalue(name)
    y_chunks, x_chunks = product.raw.chunks[-2], product.raw.chunks[-1]

    assert max(y_chunks) <= CHUNK_PX
    assert max(x_chunks) <= CHUNK_PX
    # the regression this guards against shows up as a single huge chunk
    # spanning the whole axis -- these fixtures are all well over one tile.
    assert len(y_chunks) > 1
    assert len(x_chunks) > 1


def test_sentinel2_reindexed_bands_stay_tile_chunked(sentinel2_l1c):
    # sentinel2's raw uses match= (nearest-neighbour reindex_like) to align
    # bands shot at different native resolutions -- confirm that path keeps
    # the tile chunking too, not just the single-resolution sentinel1 case.
    y_chunks, x_chunks = sentinel2_l1c.raw.chunks[-2], sentinel2_l1c.raw.chunks[-1]
    assert max(y_chunks) <= CHUNK_PX
    assert max(x_chunks) <= CHUNK_PX


@pytest.mark.parametrize("name", PRODUCTS)
def test_open_raw_chunks_to_the_given_tile_size(name):
    # _open_raw(tile) -- what patches()/patches_at() call -- opens chunked to
    # exactly the tile argument, not product.raw's own (unrelated,
    # CHUNK_PX-chunked) default.
    product = _fresh(name)
    raw = product._open_raw(256)
    try:
        y_chunks, x_chunks = raw.chunks[-2], raw.chunks[-1]
        assert max(y_chunks) <= 256
        assert max(x_chunks) <= 256
    finally:
        raw.close()
    assert "raw" not in product.__dict__, "_open_raw must not touch product.raw"
    product._close()


@pytest.mark.parametrize("name", PRODUCTS)
def test_visual_builds_once_per_loop_at_the_tile_size(name):
    product = _fresh(name)
    calls = []
    cls = type(product)
    original = cls._render_visual

    def spy(self, raw, tile):
        calls.append(tile)
        return original(self, raw, tile)

    cls._render_visual = spy
    try:
        patches = list(product.patches_at([(600, 600), (700, 700)], 256))
        assert calls == [256], "visual should build once, for this loop's own tile"
        _ = patches[0].visual
        _ = patches[1].visual  # second patch of the same loop: no rebuild
        assert calls == [256]
    finally:
        cls._render_visual = original
        product._close()
