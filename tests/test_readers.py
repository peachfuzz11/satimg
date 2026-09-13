"""``merge_bands`` must open each band pre-chunked into tiles, not collapsed to
one whole-band chunk.

A single chunk defeats every later windowed read: :meth:`Product._align_view_
chunks` rechunks ``raw`` / ``visual`` to the loop's patch size once iteration
starts, and rechunking a *single* dask chunk into tiles still depends on one
task that reads the entire band -- so every ``patch.raw`` / ``patch.visual``
access would re-decode the whole scene instead of just its own tile. See
``satimg.readers.merge_bands``'s docstring.
"""

import pytest

from satimg.readers import _TILE

PRODUCTS = ["sentinel1_iw", "sentinel2_l1c", "landsat"]


@pytest.mark.parametrize("name", PRODUCTS)
def test_raw_bands_are_tile_chunked_not_one_chunk(request, name):
    product = request.getfixturevalue(name)
    y_chunks, x_chunks = product.raw.chunks[-2], product.raw.chunks[-1]

    assert max(y_chunks) <= _TILE
    assert max(x_chunks) <= _TILE
    # the regression this guards against shows up as a single huge chunk
    # spanning the whole axis -- these fixtures are all well over one tile.
    assert len(y_chunks) > 1
    assert len(x_chunks) > 1


def test_sentinel2_reindexed_bands_stay_tile_chunked(sentinel2_l1c):
    # sentinel2's raw uses match= (nearest-neighbour reindex_like) to align
    # bands shot at different native resolutions -- confirm that path keeps
    # the tile chunking too, not just the single-resolution sentinel1 case.
    y_chunks, x_chunks = sentinel2_l1c.raw.chunks[-2], sentinel2_l1c.raw.chunks[-1]
    assert max(y_chunks) <= _TILE
    assert max(x_chunks) <= _TILE
