"""Behaviour shared by every concrete product, run against the minified test
data (pixels are zeroed; geometry, CRS and metadata are real)."""

import datetime
import itertools

import numpy
import pytest
import xarray

from satimg import sar_utils
from satimg.geometry import Window
from satimg.products.landsat import BANDS as _LS_BANDS
from satimg.tiling import read_window

PRODUCTS = ["sentinel1_iw", "sentinel2_l1c", "landsat"]


@pytest.fixture(params=PRODUCTS)
def product(request):
    return request.getfixturevalue(request.param)


_S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
             "B08", "B8A", "B09", "B10", "B11", "B12"]


class TestViews:
    def test_raw_is_band_first_dataarray(self, product):
        raw = product.raw
        assert isinstance(raw, xarray.DataArray)
        assert raw.dims == ("band", "y", "x")
        assert product.shape == (product.bands, product.height, product.width)

    @pytest.mark.parametrize(
        "name,bands",
        [("sentinel1_iw", 1), ("sentinel2_l1c", 3), ("landsat", 3)],
    )
    def test_visual_is_uint8_with_sensor_band_count(self, request, name, bands):
        product = request.getfixturevalue(name)
        visual = product.visual
        assert visual.dtype == numpy.uint8
        assert visual.sizes["band"] == bands
        assert (visual.sizes["x"], visual.sizes["y"]) == (product.width, product.height)

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("sentinel1_iw", ["VV", "VH"]),
            ("sentinel2_l1c", _S2_BANDS),
            ("landsat", list(_LS_BANDS)),
        ],
    )
    def test_raw_band_coord_is_named(self, request, name, expected):
        raw = request.getfixturevalue(name).raw
        assert [str(b) for b in raw.band.values] == expected
        # named selection works and round-trips a coord
        assert raw.sel(band=expected[0]).shape == (raw.sizes["y"], raw.sizes["x"])

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("sentinel1_iw", ["amplitude"]),
            ("sentinel2_l1c", ["red", "green", "blue"]),
            ("landsat", ["red", "green", "blue"]),
        ],
    )
    def test_visual_band_coord_is_named(self, request, name, expected):
        assert [str(b) for b in request.getfixturevalue(name).visual.band.values] == expected

    def test_sentinel2_wavelength_coord(self, sentinel2_l1c):
        wl = sentinel2_l1c.raw.wavelength_nm
        assert wl.dims == ("band",) and wl.size == 13
        assert float(wl.sel(band="B08")) == pytest.approx(832.8, abs=0.1)
        assert float(wl.sel(band="B04")) == pytest.approx(664.6, abs=0.1)


class TestPatches:
    def test_iter_yields_patches_with_origin(self, product):
        first = next(iter(product))
        assert first.window.col == 0 and first.window.row == 0
        assert first.raw.values.shape[0] == product.bands

    def test_patches_conserve_indices(self, product):
        seen = list(product.patches(512))
        assert seen[0].bounds == (0, 0, min(512, product.width), min(512, product.height))
        # every window lies inside the image and is placed on the stride grid
        for p in seen:
            assert 0 <= p.col < product.width
            assert 0 <= p.row < product.height
            assert p.col % 512 == 0 and p.row % 512 == 0

    def test_padded_patches_have_constant_shape(self, product):
        # bounded: full-res test scenes have thousands of tiles
        chan = product.bands
        head = itertools.islice(product.patches(256, edge="pad"), 8)
        assert {p.raw.values.shape for p in head} == {(chan, 256, 256)}

    def test_overlap_stride(self, product):
        chan = product.bands
        patches = list(itertools.islice(product.patches(256, overlap=32, edge="pad"), 2))
        assert patches[0].raw.values.shape == (chan, 256, 256)
        assert patches[1].window.col == 224  # stride = 256 - 32

    def test_raw_patch_matches_direct_read(self, product):
        win = Window(0, 0, 64, 64)
        patch = next(iter(product.patches(64)))
        numpy.testing.assert_array_equal(
            patch.raw.values, read_window(product.raw, win).values
        )

    def test_patches_at_keeps_the_point_centred_including_at_a_padded_edge(self, product):
        # a point one pixel in from the top-left: raw and visual must still be
        # full-size (left/top zero-padded) with the point on the centre pixel.
        col, row = 1, 1
        p = next(product.patches_at([(col, row)], 128))
        assert p.raw.shape == (product.bands, 128, 128)
        assert p.visual.shape[1:] == (128, 128)
        assert p.window.col + 64 == col and p.window.row + 64 == row
        numpy.testing.assert_array_equal(
            p.raw.values, read_window(product.raw, p.window).values
        )
        numpy.testing.assert_array_equal(
            p.visual.values, read_window(product.visual, p.window).values
        )
        # the geo-centre of the patch is the requested pixel, not some rounded
        # window centre drifting off it
        want = product.transformer.rowcol_to_latlon((row, col))
        pc_col, pc_row = p.window.center
        got = product.transformer.rowcol_to_latlon((pc_row, pc_col))
        assert (float(got[0, 0]), float(got[0, 1])) == pytest.approx(
            (float(want[0, 0]), float(want[0, 1])), abs=1e-9
        )

    def test_patch_exposes_raw_visual_and_meta(self, product):
        p = next(product.patches_at([(600, 600)], 256))
        assert p.raw.dims == ("band", "y", "x")
        assert p.raw.shape == (product.bands, 256, 256)
        assert list(p.raw.band.values) == list(product.raw.band.values)
        assert p.visual.dtype == numpy.uint8
        assert p.visual.shape[1:] == (256, 256)
        assert p.visual.sizes["band"] == product.visual.sizes["band"]
        # window is shared: raw patch == direct read at the same window
        numpy.testing.assert_array_equal(
            p.raw.values, read_window(product.raw, p.window).values
        )
        assert set(p.meta.sample()) == set(product.metadata.fields)


class TestMetadata:
    def test_timestamp(self, product):
        assert isinstance(product.timestamp, datetime.datetime)

    def test_footprint_and_bounds(self, product):
        assert product.footprint["geometry"]["type"] == "Polygon"
        min_lon, min_lat, max_lon, max_lat = product.bounds()
        assert min_lon < max_lon and min_lat < max_lat

    def test_transformer_roundtrip(self, product):
        rowcol = numpy.array([[10.0, 10.0], [100.0, 250.0]])
        latlon = product.transformer.rowcol_to_latlon(rowcol)
        back = product.transformer.latlon_to_rowcol(latlon)
        numpy.testing.assert_allclose(back, rowcol, atol=3)

    def test_patch_center_latlon_in_footprint_bbox(self, product):
        min_lon, min_lat, max_lon, max_lat = product.bounds()
        patch = next(iter(product))
        col, row = patch.window.center
        lat, lon = product.transformer.rowcol_to_latlon((row, col))[0]
        pad = 0.5
        assert min_lat - pad <= lat <= max_lat + pad
        assert min_lon - pad <= lon <= max_lon + pad


def test_sentinel1_mode(sentinel1_iw):
    assert sentinel1_iw.mode == "IW"


def _local_platform_heading(product, rowcol):
    lat = product.transformer.rowcol_to_latlon(rowcol)[:, 0]
    ascending = product.metadata.attrs["pass"].lower() == "ascending"
    return sar_utils.ground_track_heading(lat, product.metadata.attrs["orbit_inclination"], ascending)[0]


def test_sentinel1_heading_to_los(sentinel1_iw):
    rowcol = (sentinel1_iw.height // 2, sentinel1_iw.width // 2)
    look_direction = (_local_platform_heading(sentinel1_iw, rowcol) + 90.0) % 360.0
    assert sentinel1_iw.heading_to_los(rowcol, look_direction) == pytest.approx(0.0)
    assert sentinel1_iw.heading_to_los(rowcol, (look_direction + 180.0) % 360.0) == pytest.approx(-180.0)


def test_sentinel1_doppler_azimuth_shift_signs(sentinel1_iw):
    rowcol = (sentinel1_iw.height // 2, sentinel1_iw.width // 2)
    look_direction = (_local_platform_heading(sentinel1_iw, rowcol) + 90.0) % 360.0

    away = sentinel1_iw.doppler_azimuth_shift(rowcol, 10.0, look_direction)
    toward = sentinel1_iw.doppler_azimuth_shift(rowcol, 10.0, (look_direction + 180.0) % 360.0)
    along_track = sentinel1_iw.doppler_azimuth_shift(rowcol, 10.0, (look_direction + 90.0) % 360.0)

    assert away > 0
    assert toward < 0
    assert along_track == pytest.approx(0.0, abs=1e-6)
    assert away == pytest.approx(-toward)


def test_sentinel1_heading_in_image(sentinel1_iw):
    rowcol = (sentinel1_iw.height // 2, sentinel1_iw.width // 2)
    up_heading = (_local_platform_heading(sentinel1_iw, rowcol) + 180.0) % 360.0
    assert sentinel1_iw.heading_in_image(rowcol, up_heading) == pytest.approx(0.0)
    assert sentinel1_iw.heading_in_image(rowcol, (up_heading + 90.0) % 360.0) == pytest.approx(90.0)


def test_sentinel1_descending_pass_is_roughly_north_up(sentinel1_iw):
    # sanity check against the well-known SAR fact: a descending-pass GRD scene
    # is displayed close to north-up (ascending passes are the opposite, south-up)
    assert sentinel1_iw.metadata.attrs["pass"].lower() == "descending"
    rowcol = (sentinel1_iw.height // 2, sentinel1_iw.width // 2)
    got = sentinel1_iw.heading_in_image(rowcol, 0.0)
    assert got < 20.0 or got > 340.0


def test_sentinel1_correct_position(sentinel1_iw):
    rowcol = (sentinel1_iw.height // 2, sentinel1_iw.width // 2)
    lat, lon = sentinel1_iw.transformer.rowcol_to_latlon(rowcol)[0]
    look_direction = (_local_platform_heading(sentinel1_iw, rowcol) + 90.0) % 360.0

    shift_px = sentinel1_iw.doppler_azimuth_shift(rowcol, 10.0, look_direction)
    corrected_lat, corrected_lon = sentinel1_iw.correct_position(lat, lon, 10.0, look_direction)
    corrected_rowcol = sentinel1_iw.transformer.latlon_to_rowcol((corrected_lat, corrected_lon))[0]

    # GCPTransformer's polynomial GCP fit isn't an exact sub-pixel round-trip,
    # so check against a 1-pixel tolerance rather than the raw float shift.
    assert corrected_rowcol[0] == pytest.approx(rowcol[0] - shift_px, abs=1.0)
    assert corrected_rowcol[1] == pytest.approx(rowcol[1], abs=1.0)

    # along-track motion produces no azimuth shift, so no correction either
    along_track = (look_direction + 90.0) % 360.0
    unmoved_lat, unmoved_lon = sentinel1_iw.correct_position(lat, lon, 10.0, along_track)
    assert unmoved_lat == pytest.approx(lat, abs=1e-6)
    assert unmoved_lon == pytest.approx(lon, abs=1e-6)


def test_thumbnail_opens(product):
    thumb = product.thumbnail()
    assert thumb.size == (200, 200)
