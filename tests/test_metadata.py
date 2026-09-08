"""Per-pixel metadata: the bilinear kernel, the three product readers, the
full-grid DataArray, and ``patch.meta``.

Run against the minified scenes. ``tests/minify.py`` zeroes every ``.TIF``, so
Landsat angle *values* are all 0.0 here -- Landsat assertions cover plumbing
(fields / units / shape / clamping) only. Sentinel-1 and Sentinel-2 keep their
real XML, so those get real numbers.
"""

import dask.array
import numpy
import pytest
import xarray

import satimg
from satimg.geometry import Window
from satimg.metadata import Metadata, bilinear
from satimg.product import Product
from satimg.tiling import Patch, read_window


# -- the kernel --------------------------------------------------------

class TestBilinear:
    rows = numpy.array([0.0, 10.0, 30.0])
    cols = numpy.array([0.0, 5.0, 25.0])
    grid = numpy.array([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0], [30.0, 31.0, 32.0]])

    def test_exact_at_nodes(self):
        for i, r in enumerate(self.rows):
            for j, c in enumerate(self.cols):
                got = bilinear(self.rows, self.cols, self.grid, r, c)
                assert got == pytest.approx(self.grid[i, j])

    def test_midpoint_is_corner_mean(self):
        got = bilinear(self.rows, self.cols, self.grid, 5.0, 2.5)
        assert got == pytest.approx((0 + 1 + 10 + 11) / 4)

    def test_out_of_range_clamps_to_edge(self):
        assert bilinear(self.rows, self.cols, self.grid, -100, -100) == pytest.approx(0.0)
        assert bilinear(self.rows, self.cols, self.grid, 1e9, 1e9) == pytest.approx(32.0)

    def test_nan_corner_propagates(self):
        g = self.grid.copy()
        g[0, 0] = numpy.nan
        assert numpy.isnan(bilinear(self.rows, self.cols, g, 2.0, 1.0))
        # a cell away from the NaN corner is still fine
        assert not numpy.isnan(bilinear(self.rows, self.cols, g, 20.0, 20.0))

    def test_vectorised(self):
        qr = numpy.array([0.0, 10.0, 30.0])
        qc = numpy.array([0.0, 5.0, 25.0])
        got = bilinear(self.rows, self.cols, self.grid, qr, qc)
        numpy.testing.assert_allclose(got, [0.0, 11.0, 32.0])


# -- Sentinel-1 -------------------------------------------------------

class TestSentinel1:
    def test_fields(self, sentinel1_iw):
        assert sentinel1_iw.metadata.fields == [
            "incidence_angle", "elevation_angle", "slant_range_time", "height",
        ]

    def test_incidence_rises_across_range(self, sentinel1_iw):
        m = sentinel1_iw.metadata
        near = m.incidence_angle.at((0, 0))
        far = m.incidence_angle.at((0, sentinel1_iw.width - 1))
        assert 25 < near < 40 < far < 50          # near-range -> far-range
        assert m.attrs["incidence_angle_mid_swath"] == pytest.approx((near + far) / 2, abs=3)

    def test_slant_range_time_scale(self, sentinel1_iw):
        t = sentinel1_iw.metadata.slant_range_time.at((8000, 12000))
        assert 1e-3 < t < 1e-2                     # ~5 ms two-way delay

    def test_attrs(self, sentinel1_iw):
        assert sentinel1_iw.metadata.attrs["pass"].lower() in {"ascending", "descending"}

    def test_units(self, sentinel1_iw):
        m = sentinel1_iw.metadata
        assert m.incidence_angle.units == "degrees"
        assert m.slant_range_time.units == "seconds"


# -- Sentinel-2 -------------------------------------------------------

class TestSentinel2:
    def test_fields(self, sentinel2_l1c):
        assert sentinel2_l1c.metadata.fields == [
            "sun_zenith", "sun_azimuth", "view_zenith", "view_azimuth",
        ]

    def test_sun_zenith_in_grid_range(self, sentinel2_l1c):
        z = sentinel2_l1c.metadata.sun_zenith
        assert z.at((0, 0)) == pytest.approx(z._values[0, 0], abs=1e-6)
        assert 60 < z.at((5000, 5000)) < 85

    def test_viewing_grid_has_no_nan_after_fill(self, sentinel2_l1c):
        for name in ("view_zenith", "view_azimuth"):
            assert not numpy.isnan(sentinel2_l1c.metadata[name]._values).any()

    def test_attrs_match_mean(self, sentinel2_l1c):
        m = sentinel2_l1c.metadata
        grid_mean = float(numpy.nanmean(m.sun_zenith._values))
        assert m.attrs["mean_sun_zenith"] == pytest.approx(grid_mean, abs=1.0)


# -- Landsat --------------------------------------------------------

class TestLandsat:
    def test_fields_and_units(self, landsat):
        m = landsat.metadata
        assert m.fields == ["sun_zenith", "sun_azimuth", "view_zenith", "view_azimuth"]
        assert all(m[f].units == "degrees" for f in m.fields)

    def test_at_returns_float_and_clamps(self, landsat):
        m = landsat.metadata
        val = m.sun_zenith.at((landsat.height // 2, landsat.width // 2))
        assert isinstance(val, float)
        # far outside the grid must clamp, not raise
        m.sun_zenith.at((-9999, -9999))
        m.sun_zenith.at((landsat.height + 9999, landsat.width + 9999))

    def test_attrs_are_numeric(self, landsat):
        attrs = landsat.metadata.attrs
        assert isinstance(attrs["sun_elevation"], float)
        assert 0 < attrs["earth_sun_distance"] < 2


# -- Field.grid --------------------------------------------------

def test_field_grid_is_lazy_full_grid(sentinel1_iw):
    field = sentinel1_iw.metadata.incidence_angle
    grid = field.grid
    assert isinstance(grid, xarray.DataArray)
    assert grid.dims == ("y", "x")
    assert grid.shape == (sentinel1_iw.height, sentinel1_iw.width)
    assert grid.dtype == numpy.float64
    assert isinstance(grid.data, dask.array.Array)                  # not materialised

    win = Window(0, 0, 64, 64)
    sub = read_window(grid, win)
    assert tuple(sub.shape) == (64, 64)
    assert float(sub.values[0, 0]) == pytest.approx(field.at((0, 0)), abs=1e-6)


def test_field_patches_carry_transform(sentinel1_iw):
    field = sentinel1_iw.metadata.incidence_angle
    patch = next(iter(field.patches(512)))
    assert patch.center_latlon                                      # transform attached


# -- patch.meta --------------------------------------------------

def test_patch_meta_whole_window_and_point(sentinel1_iw):
    m = sentinel1_iw.metadata
    for patch in sentinel1_iw.patches(256):
        whole = patch.meta.incidence_angle
        assert tuple(whole.shape) == (256, 256)
        assert isinstance(whole.data, dask.array.Array)             # lazy

        # patch-local (0, 0) == product-global (window.row, window.col)
        local = patch.meta.at((0, 0))["incidence_angle"]
        glob = m.incidence_angle.at((patch.window.row, patch.window.col))
        assert local == pytest.approx(glob, abs=1e-6)

        sample = patch.meta.sample()
        assert set(sample) == set(m.fields)
        assert all(isinstance(v, float) for v in sample.values())
        break


def test_patch_meta_getitem(sentinel2_l1c):
    patch = next(iter(sentinel2_l1c.patches(128)))
    assert tuple(patch.meta["sun_zenith"].shape) == (128, 128)


# -- corners() -------------------------------------------------

_CORNER_KEYS = {"top_left", "top_right", "bottom_left", "bottom_right", "center"}


def test_field_corners_full_grid(sentinel1_iw):
    c = sentinel1_iw.metadata.incidence_angle.corners()
    assert set(c) == _CORNER_KEYS
    assert all(isinstance(v, float) for v in c.values())
    # incidence rises left -> right across the range direction
    assert c["top_left"] < c["center"] < c["top_right"]
    assert c["bottom_left"] < c["center"] < c["bottom_right"]
    assert c["center"] == pytest.approx(
        sentinel1_iw.metadata.incidence_angle.at(
            (sentinel1_iw.height / 2, sentinel1_iw.width / 2)
        )
    )


def test_metadata_corners_every_field(sentinel2_l1c):
    mc = sentinel2_l1c.metadata.corners()
    assert set(mc) == set(sentinel2_l1c.metadata.fields)
    for c in mc.values():
        assert set(c) == _CORNER_KEYS


def test_patch_meta_corners(sentinel1_iw):
    patch = next(sentinel1_iw.patches_at([(6000, 9000)], 256))
    pc = patch.meta.corners()
    assert set(pc) == set(sentinel1_iw.metadata.fields)
    inc = pc["incidence_angle"]
    assert set(inc) == _CORNER_KEYS
    assert inc["top_left"] < inc["top_right"]        # near -> far range across the tile
    assert inc["center"] == pytest.approx(patch.meta.sample()["incidence_angle"])


def test_field_corners_without_grid_raises():
    from satimg.metadata import Field

    bare = Field([0, 1], [0, 1], [[0.0, 1.0], [2.0, 3.0]], name="x")
    with pytest.raises(AttributeError):
        bare.corners()


def test_bare_patch_has_no_meta():
    da = xarray.DataArray(numpy.zeros((1, 8, 8)), dims=("band", "y", "x"))
    patch = Patch(da, Window(0, 0, 4, 4))
    with pytest.raises(AttributeError):
        patch.meta


# -- default (no reader) ------------------------------------------

def test_product_without_reader_has_empty_metadata():
    class Bare(Product):
        raw = property(lambda self: None)
        transformer = property(lambda self: None)
        timestamp = property(lambda self: None)
        footprint = property(lambda self: None)

        def _render_visual(self):  # pragma: no cover - not exercised
            raise NotImplementedError

        def thumbnail(self):  # pragma: no cover
            raise NotImplementedError

    meta = Bare("/nowhere").metadata
    assert isinstance(meta, Metadata)
    assert meta.fields == []
    assert repr(meta) == "Metadata(empty)"
