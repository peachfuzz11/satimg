import json
import zipfile

import numpy
import pytest
from rasterio.crs import CRS
from rasterio.transform import Affine

import satimg
from satimg import NisarGeometry
from satimg.connectors import NisarConnector, get_connector
from satimg.connectors._credentials import CredentialsError
from satimg.connectors.nisar import _wkt
from satimg.products import NisarGCOVProduct
from satimg.products.nisar.files import parse_polygon, parse_time
from satimg.registry import resolve
from satimg.transform import AffineTransformer
from tests import nisar_fixture as fx

SPEED = 10.0  # m/s


@pytest.fixture(scope="module")
def h5_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("nisar") / fx.NAME
    fx.build(path)
    return path


@pytest.fixture(scope="module")
def product(h5_path):
    with satimg.open(str(h5_path)) as p:
        yield p


# -- parsing ----------------------------------------------------------------
def test_parse_time_keeps_microseconds_of_nanoseconds():
    t = parse_time("2026-08-01T06:10:25.123456789")
    assert (t.second, t.microsecond) == (25, 123456)
    assert parse_time("2026-08-01T06:10:25").microsecond == 0


def test_parse_polygon_drops_height_and_closes():
    feature = parse_polygon("POLYGON Z ((1 2 0, 3 2 0, 3 4 0))")
    assert feature["geometry"]["coordinates"][0] == [(1, 2), (3, 2), (3, 4), (1, 2)]


@pytest.mark.parametrize("name", [
    fx.NAME,
    fx.NAME[:-3],  # a directory / zip of the same name
    "NISAR_L2_UR_GCOV_035_005_D_083_4005_DHDH_A_20261105T132813_20261105T132832_P05023_F_P_J_001.h5",
])
def test_resolves(name):
    assert resolve(name) is NisarGCOVProduct


# -- product ------------------------------------------------------------------
def test_opens_a_bare_h5(product):
    assert isinstance(product, NisarGCOVProduct)
    assert product.shape == (2, fx.HEIGHT, fx.WIDTH)
    assert list(product.raw.band.values) == ["HHHH", "HVHV"]
    assert product.dtype == numpy.float32
    assert product.timestamp.isoformat() == "2026-08-01T06:10:17.750000"
    assert product.metadata.attrs["look_direction"] == "left"
    assert product.metadata.attrs["pass"] == "descending"
    assert product.metadata.attrs["platform_velocity"] == pytest.approx(fx.PLATFORM_SPEED)
    assert product.bounds() == pytest.approx((14.8, 68.47, 14.95, 68.55))


def test_raw_is_linear_power_with_nan_outside_the_swath(product):
    hh = product.raw.sel(band="HHHH").values
    assert hh[fx.SHIP] == pytest.approx(fx.SHIP_POWER)
    assert hh[0, 100] == pytest.approx(fx.SEA_POWER)
    assert numpy.isnan(hh[:, :10]).all()


def test_visual_ship_bright_sea_dark_nodata_black(product):
    v = product.visual.values[0]
    assert product.visual.dtype == numpy.uint8
    assert v[fx.SHIP] > 240
    assert v[0, 100] < 40
    assert (v[:, :10] == 0).all()


def test_patches_carry_the_ship(product):
    patch = next(p for p in product.patches_at([(fx.SHIP[1], fx.SHIP[0])], 64))
    assert patch.visual.values[0, 32, 32] > 240
    assert patch.meta.sample()["incidence_angle"] == pytest.approx(fx.INCIDENCE_0, abs=1e-3)


def test_transformer_uses_cf_pixel_centres(product):
    # pixel (0, 0)'s centre is at (X0, Y0)
    t = product.transformer
    lat, lon = t.rowcol_to_latlon((0.5, 0.5))[0]
    utm = AffineTransformer(Affine.identity(), CRS.from_epsg(fx.EPSG))
    row, col = utm.latlon_to_rowcol((lat, lon))[0]  # identity: row = y, col = x
    assert (col, row) == pytest.approx((fx.X0, fx.Y0), abs=1e-3)
    assert t.latlon_to_rowcol((lat, lon))[0] == pytest.approx((0.5, 0.5), abs=1e-6)
    # raw carries the same grid: map coordinates of pixel centres
    assert product.raw.rio.crs.to_epsg() == fx.EPSG
    assert (float(product.raw.x[0]), float(product.raw.y[0])) == (fx.X0, fx.Y0)


def test_metadata_is_the_cube_at_sea_level(product):
    m = product.metadata
    assert set(m.fields) >= {"incidence_angle", "slant_range", "los_east", "along_track_north"}
    centre = (fx.HEIGHT / 2, fx.WIDTH / 2)
    assert m.incidence_angle.at(centre) == pytest.approx(fx.INCIDENCE_0, abs=1e-4)
    assert m.slant_range.at(centre) == pytest.approx(fx.SLANT_RANGE_0)
    assert m.ground_track_velocity.at(centre) == pytest.approx(6_900.0)


def test_thumbnail(product):
    assert product.thumbnail().size == satimg.readers.THUMBNAIL_SIZE


def test_zip_native(h5_path, tmp_path):
    archive = tmp_path / (fx.NAME[:-3] + ".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as z:
        z.write(h5_path, fx.NAME)
    with satimg.open(str(archive)) as p:
        assert isinstance(p, NisarGCOVProduct)
        assert p.raw.sel(band="HHHH").values[fx.SHIP] == pytest.approx(fx.SHIP_POWER)


# -- SAR geometry ---------------------------------------------------------------
CENTRE = (fx.HEIGHT / 2, fx.WIDTH / 2)


def test_heading_to_los_follows_the_left_look(product):
    assert product.heading_to_los(CENTRE, fx.LOOK) == pytest.approx(0.0, abs=1e-3)
    assert abs(product.heading_to_los(CENTRE, fx.PLATFORM_HEADING)) == pytest.approx(90.0, abs=1e-3)


def test_no_shift_moving_along_track(product):
    assert product.doppler_azimuth_shift_m(CENTRE, SPEED, fx.PLATFORM_HEADING) == pytest.approx(0.0, abs=1e-3)


def test_closing_target_shifts_forward_by_the_textbook_amount(product):
    towards_satellite = (fx.LOOK + 180) % 360
    shift = product.doppler_azimuth_shift_m(CENTRE, SPEED, towards_satellite)
    expected = SPEED * numpy.sin(numpy.deg2rad(fx.INCIDENCE_0)) * fx.SLANT_RANGE_0 / fx.PLATFORM_SPEED
    assert shift == pytest.approx(expected, rel=1e-4)
    # receding: the same, backwards
    assert product.doppler_azimuth_shift_m(CENTRE, SPEED, fx.LOOK) == pytest.approx(-expected, rel=1e-4)


def test_pixel_shift_points_along_the_ground_track(product):
    towards_satellite = (fx.LOOK + 180) % 360
    d_row, d_col = product.doppler_pixel_shift(CENTRE, SPEED, towards_satellite)
    shift_px = product.doppler_azimuth_shift_m(CENTRE, SPEED, towards_satellite) / fx.SPACING
    assert numpy.hypot(d_row, d_col) == pytest.approx(shift_px, rel=1e-3)
    # flying 192 degrees: down the image, a little left (plus UTM convergence)
    direction = numpy.rad2deg(numpy.arctan2(d_col, -d_row)) % 360
    assert direction == pytest.approx(product.heading_in_image(CENTRE, fx.PLATFORM_HEADING), abs=1e-6)
    assert d_row > 0 and d_col < 0


def test_correct_position_undoes_the_shift(product):
    towards_satellite = (fx.LOOK + 180) % 360
    true_lat, true_lon = product.transformer.rowcol_to_latlon(CENTRE)[0]
    shift = product.doppler_pixel_shift(CENTRE, SPEED, towards_satellite)
    seen_lat, seen_lon = product.transformer.rowcol_to_latlon(numpy.add(CENTRE, shift))[0]
    lat, lon = product.correct_position(seen_lat, seen_lon, SPEED, towards_satellite)
    assert (lat, lon) == pytest.approx((true_lat, true_lon), abs=1e-6)


def test_heading_round_trip_and_convergence(product):
    # ~1.2 degrees west of zone 33's 15 E meridian at 68.5 N: grid convergence
    # of about 1.2 * sin(68.5) = 1.1 degrees, true north clockwise of grid north
    image = product.heading_in_image(CENTRE, 45.0)
    assert image - 45.0 == pytest.approx(1.1, abs=0.1)
    assert product.heading_from_image(CENTRE, image) == pytest.approx(45.0)


def test_north_angle_on_a_polar_grid():
    # EPSG:3413, NSIDC polar stereographic north: true north points at the pole
    t = AffineTransformer(Affine(100.0, 0, 0, 0, -100.0, 0), CRS.from_epsg(3413))
    # x > 0, y = 0 is on the 135 W meridian... pole up and to the left of (row 0, col 10_000)
    angle = t.north_angle((0.0, 10_000.0))[0]
    assert angle == pytest.approx(270.0, abs=0.01)


def test_geometry_round_trips_and_keeps_sar_methods(product):
    stored = json.loads(json.dumps(product.geometry.to_dict(), allow_nan=False))
    assert len(json.dumps(stored)) < 200_000
    geometry = NisarGeometry.from_dict(stored)
    towards_satellite = (fx.LOOK + 180) % 360
    assert geometry.doppler_azimuth_shift_m(CENTRE, SPEED, towards_satellite) == pytest.approx(
        product.doppler_azimuth_shift_m(CENTRE, SPEED, towards_satellite), rel=1e-6
    )
    assert geometry.correct_position(68.5, 14.9, SPEED, 0.0) == pytest.approx(
        product.correct_position(68.5, 14.9, SPEED, 0.0), abs=1e-7
    )


# -- connector ------------------------------------------------------------------
def test_connector_factory_is_offline():
    assert isinstance(get_connector("nisar-gcov"), NisarConnector)


def test_connector_wkt():
    polygon = {"type": "Polygon", "coordinates": [[[1, 2], [3, 2], [3, 4], [1, 2]]]}
    assert _wkt({"type": "Feature", "geometry": polygon}) == "POLYGON((1 2, 3 2, 3 4, 1 2))"
    assert _wkt({"type": "Point", "coordinates": [5, 6]}) == "POINT(5 6)"


def test_connector_needs_a_token_or_login(monkeypatch):
    for var in ("EARTHDATA_TOKEN", "EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(CredentialsError, match="EARTHDATA_TOKEN"):
        NisarConnector()._get_session()
    session = NisarConnector(token="t")._get_session()
    assert session.headers["Authorization"] == "Bearer t"
    assert NisarConnector(username="u", password="p")._get_session().auth == ("u", "p")
