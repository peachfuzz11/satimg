"""The moving-target correction and heading conversions every SAR product
shares (:class:`~satimg.sar_product.SARProduct`, on
``satimg/sar_orbit_geometry.py``): Sentinel-1 (real minified IW and EW
products), NISAR, Umbra, Capella, ICEYE. The NISAR, Capella and ICEYE
fixtures fly a straight broadside orbit (``orbit_fixture``), for which the
textbook shift is exact; Umbra's orbit is rebuilt from its scene-centre
geometry."""

import json

import numpy
import pytest

import satimg
from satimg import sar
from satimg.sar_orbit_geometry import SAROrbitGeometry
from satimg.sar_product import SARProduct
from tests import capella_fixture, iceye_fixture, nisar_fixture, orbit_fixture, umbra_fixture
from tests.conftest import PRODUCT_PATHS

#: m/s -- slow enough that the shift stays inside the small fixture images
SPEED = 3.0

FIXTURES = {"capella": capella_fixture, "iceye": iceye_fixture}

#: what every SAR product answers to
INTERFACE = (
    "heading_to_los", "heading_in_image", "heading_from_image",
    "doppler_azimuth_shift_m", "doppler_pixel_shift", "correct_position",
)


@pytest.fixture(scope="module", params=["sentinel1_iw", "sentinel1_ew", "nisar", "umbra", "capella", "iceye"])
def product(request, tmp_path_factory):
    if request.param in PRODUCT_PATHS:
        path = PRODUCT_PATHS[request.param]
    elif request.param == "nisar":
        path = tmp_path_factory.mktemp("nisar") / nisar_fixture.NAME
        nisar_fixture.build(path)
    else:
        fx = {"umbra": umbra_fixture, **FIXTURES}[request.param]
        path = fx.build(tmp_path_factory.mktemp(request.param))
    with satimg.open(str(path)) as p:
        yield p


def _middle(product):
    return ((product.height - 1) / 2, (product.width - 1) / 2)


def _look(product, rowcol):
    """The compass look bearing at ``rowcol`` (heading_to_los == 0)."""
    return (0.0 - product.heading_to_los(rowcol, 0.0)) % 360.0


def test_every_sar_product_has_the_same_interface():
    sar_products = [cls for cls in vars(satimg).values() if isinstance(cls, type) and issubclass(cls, SARProduct)]
    assert {cls.__name__ for cls in sar_products} == {
        "SARProduct", "Sentinel1GRD", "NisarGCOV", "UmbraGEC", "CapellaGEC", "IceyeGRD"
    }
    for cls in sar_products:
        for name in INTERFACE:
            assert getattr(cls, name) is getattr(SARProduct, name), (cls, name)


def test_geometry_is_an_orbit_geometry(product):
    assert isinstance(product, SARProduct)
    assert isinstance(product.geometry, SAROrbitGeometry)
    assert set(product.metadata.attrs["orbit"]) == {"epoch", "times", "positions", "velocities"}


def test_along_track_motion_does_not_shift(product):
    rc = _middle(product)
    along = _look(product, rc) + 90.0
    assert product.doppler_azimuth_shift_m(rc, SPEED, along) == pytest.approx(0.0, abs=0.05)


def test_closing_target_shifts_forward_receding_back(product):
    rc = _middle(product)
    towards = (_look(product, rc) + 180.0) % 360.0
    closing = product.doppler_azimuth_shift_m(rc, SPEED, towards)
    # tens of metres per m/s, for an R / V of 80-100 s
    assert 50.0 < closing < 300.0
    assert product.doppler_azimuth_shift_m(rc, SPEED, _look(product, rc)) == pytest.approx(-closing, rel=1e-6)


@pytest.mark.parametrize("name", list(FIXTURES))
def test_textbook_shift_on_a_broadside_orbit(name, tmp_path):
    fx = FIXTURES[name]
    with satimg.open(str(fx.build(tmp_path))) as product:
        rc = product.transformer.latlon_to_rowcol(fx.TARGET)[0]
        assert product.heading_to_los(rc, fx.LOOK) == pytest.approx(0.0, abs=1e-3)
        towards = (fx.LOOK + 180.0) % 360.0
        expected = SPEED * numpy.sin(numpy.deg2rad(fx.INCIDENCE)) * fx.SLANT_RANGE / orbit_fixture.SPEED
        assert product.doppler_azimuth_shift_m(rc, SPEED, towards) == pytest.approx(expected, rel=1e-4)


def test_pixel_shift_and_correct_position_are_inverses(product):
    rc = numpy.array(_middle(product))
    heading = (_look(product, rc) + 150.0) % 360.0
    true_lat, true_lon = product.transformer.rowcol_to_latlon(rc)[0]
    shift = product.doppler_pixel_shift(rc, SPEED, heading)
    assert numpy.hypot(*shift) > 3  # pixels: well off the true position
    seen_lat, seen_lon = product.transformer.rowcol_to_latlon(rc + shift)[0]
    lat, lon = product.correct_position(seen_lat, seen_lon, SPEED, heading)
    assert (lat, lon) == pytest.approx((true_lat, true_lon), abs=1e-7)


def test_headings_round_trip(product):
    rc = _middle(product)
    for heading in (0.0, 73.0, 190.0, 300.0):
        image = product.heading_in_image(rc, heading)
        assert (product.heading_from_image(rc, image) - heading + 180) % 360 - 180 == pytest.approx(0.0, abs=1e-6)


def test_vectorised(product):
    middle = numpy.array(_middle(product))
    rcs = numpy.array([middle, middle + (-30.0, 20.0), middle + (40.0, -10.0)])
    shifts = product.doppler_azimuth_shift_m(rcs, SPEED, 33.0)
    pixels = product.doppler_pixel_shift(rcs, SPEED, 33.0)
    assert shifts.shape == (3,) and pixels.shape == (3, 2)
    for rc, shift, pixel in zip(rcs, shifts, pixels):
        assert shift == pytest.approx(product.doppler_azimuth_shift_m(tuple(rc), SPEED, 33.0))
        assert pixel == pytest.approx(product.doppler_pixel_shift(tuple(rc), SPEED, 33.0))


def test_geometry_round_trips_and_keeps_the_correction(product):
    stored = json.loads(json.dumps(product.geometry.to_dict(), allow_nan=False))
    geometry = SAROrbitGeometry.from_dict(stored)
    rc = _middle(product)
    # (ICEYE's swath transformer stores its node grid to ~1e-7)
    assert geometry.doppler_azimuth_shift_m(rc, SPEED, 70.0) == pytest.approx(
        product.doppler_azimuth_shift_m(rc, SPEED, 70.0), rel=1e-5
    )
    lat, lon = product.transformer.rowcol_to_latlon(rc)[0]
    assert geometry.correct_position(lat, lon, SPEED, 70.0) == pytest.approx(
        product.correct_position(lat, lon, SPEED, 70.0), abs=1e-8
    )


def test_umbra_older_metadata_gives_the_same_orbit(tmp_path):
    (tmp_path / "stac").mkdir()
    (tmp_path / "v1").mkdir()
    with satimg.open(str(umbra_fixture.build(tmp_path / "stac"))) as stac, \
            satimg.open(str(umbra_fixture.build(tmp_path / "v1", v1=True))) as v1:
        rc = _middle(stac)
        assert v1.doppler_azimuth_shift_m(rc, SPEED, 70.0) == pytest.approx(
            stac.doppler_azimuth_shift_m(rc, SPEED, 70.0), rel=1e-3
        )


def test_umbra_rebuilt_orbit_is_earth_fixed(tmp_path):
    with satimg.open(str(umbra_fixture.build(tmp_path))) as product:
        orbit = product.metadata.attrs["orbit"]
    position, velocity = numpy.array(orbit["positions"]).mean(axis=0), numpy.array(orbit["velocities"][0])
    radius = numpy.linalg.norm(position)
    assert 6_800e3 < radius < 7_100e3  # Umbra's ~450-700 km altitude
    # Earth-fixed: a little faster than the inertial circular speed on a
    # retrograde (sun-synchronous) orbit
    inertial = numpy.sqrt(3.986004418e14 / radius)
    assert inertial < numpy.linalg.norm(velocity) < inertial * 1.02
    # the sensor is where the line of sight puts it
    lat, lon = product.transformer.rowcol_to_latlon(_middle(product))[0]
    assert numpy.linalg.norm(position - sar.geodetic_to_ecef(lat, lon)[0]) == pytest.approx(
        umbra_fixture.STAC["properties"]["umbra:slant_range_meters"], rel=1e-6
    )


def test_no_orbit_is_a_clear_error(tmp_path):
    stac = json.loads(json.dumps(umbra_fixture.STAC))
    del stac["properties"]["view:azimuth"]
    folder = umbra_fixture.build(tmp_path)
    (folder / f"{umbra_fixture.NAME}.stac.v2.json").write_text(json.dumps(stac))
    with satimg.open(str(folder)) as product:
        assert product.metadata.attrs["orbit"] is None
        with pytest.raises(ValueError, match="no sensor orbit"):
            product.correct_position(*product.transformer.rowcol_to_latlon((10, 10))[0], SPEED, 0.0)


def test_pixel_shift_is_finite_when_it_lands_off_the_image(product):
    # a fast target at a corner: shifted well off the grid, still a shift
    rc = (2.0, 2.0)
    towards = (_look(product, rc) + 180.0) % 360.0
    fast = product.doppler_pixel_shift(rc, 30.0, towards)
    slow = product.doppler_pixel_shift(rc, 1.0, towards)
    assert numpy.all(numpy.isfinite(fast))
    assert fast == pytest.approx(30.0 * slow, rel=1e-6)
