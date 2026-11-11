"""SAR-specific per-pixel geometry: heading conversion and the moving-target
azimuth-shift displacement itself. See ``satimg/sar.py`` for the
derivation and sign conventions."""

import numpy
import pytest

from satimg import sar
from satimg.sar import (
    azimuth_shift_m,
    ground_track_heading,
    heading_from_image,
    heading_in_image,
    heading_to_los,
)


class TestGroundTrackHeading:
    inclination = 98.18  # Sentinel-1's sun-synchronous orbit

    def test_ascending_and_descending_are_supplementary(self):
        lat = 45.0
        asc = ground_track_heading(lat, self.inclination, True)
        desc = ground_track_heading(lat, self.inclination, False)
        expected_desc = ((180.0 - asc) + 180.0) % 360.0 - 180.0
        assert desc == pytest.approx(expected_desc)

    def test_ascending_heading_is_west_of_north(self):
        # Sentinel-1's retrograde sun-synchronous orbit gives an ascending heading
        # a little west of due north, matching known operational values (roughly
        # -10 to -15 degrees at mid-latitudes)
        heading = ground_track_heading(45.0, self.inclination, True)
        assert -20.0 < heading < 0.0

    def test_vectorised_over_latitude(self):
        lats = numpy.array([0.0, 30.0, 60.0])
        got = ground_track_heading(lats, self.inclination, True)
        assert got.shape == (3,)


class TestHeadingToLos:
    def test_moving_away_is_zero(self):
        # look direction is platform_heading + 90; heading equal to it -> 0
        assert heading_to_los(90.0, 0.0) == pytest.approx(0.0)

    def test_moving_toward_is_minus_180(self):
        assert heading_to_los(270.0, 0.0) == pytest.approx(-180.0)

    def test_along_track_is_plus_minus_90(self):
        assert heading_to_los(0.0, 0.0) == pytest.approx(-90.0)
        assert heading_to_los(180.0, 0.0) == pytest.approx(90.0)

    def test_wraps_with_real_negative_platform_heading(self):
        # a real Sentinel-1 descending-pass platform_heading, e.g. -168.04 degrees
        platform_heading = -168.038694
        look_direction = (platform_heading + 90.0) % 360.0
        got = heading_to_los(look_direction, platform_heading)
        assert got == pytest.approx(0.0)

    def test_result_in_range(self):
        for heading in numpy.linspace(0, 360, 37, endpoint=False):
            got = heading_to_los(heading, -168.038694)
            assert -180.0 <= got < 180.0

    def test_vectorised(self):
        got = heading_to_los(numpy.array([90.0, 270.0, 0.0, 180.0]), 0.0)
        numpy.testing.assert_allclose(got, [0.0, -180.0, -90.0, 90.0])


class TestHeadingInImage:
    """GRD frame: row (image "down") runs along the platform heading, column
    (image "right") along the look direction, platform heading + 90."""

    platform_heading = -168.038694

    def test_platform_heading_points_down(self):
        got = heading_in_image(self.platform_heading, self.platform_heading)
        assert got == pytest.approx(180.0)

    def test_opposite_of_platform_heading_points_up(self):
        got = heading_in_image(self.platform_heading + 180.0, self.platform_heading)
        assert got == pytest.approx(0.0)

    def test_look_direction_points_right(self):
        got = heading_in_image(self.platform_heading + 90.0, self.platform_heading)
        assert got == pytest.approx(90.0)

    def test_away_from_look_direction_points_left(self):
        got = heading_in_image(self.platform_heading - 90.0, self.platform_heading)
        assert got == pytest.approx(270.0)

    def test_is_a_reflection_not_a_rotation(self):
        # turning the ship clockwise on the map turns it *counter*-clockwise on the raster
        step = 10.0
        for heading in numpy.linspace(0, 360, 12, endpoint=False):
            turned = heading_in_image(heading + step, self.platform_heading)
            base = heading_in_image(heading, self.platform_heading)
            assert (turned - base + 180.0) % 360.0 - 180.0 == pytest.approx(-step)

    def test_wraps_to_zero_to_360(self):
        for heading in numpy.linspace(-360, 720, 41):
            assert 0.0 <= heading_in_image(heading, self.platform_heading) < 360.0

    def test_vectorised(self):
        got = heading_in_image(numpy.array([30.0, 210.0, 120.0, -60.0]), 30.0)
        numpy.testing.assert_allclose(got, [180.0, 0.0, 90.0, 270.0])


class TestHeadingFromImage:
    @pytest.mark.parametrize("platform", [-12.3, 0.0, 192.0, 347.5])
    def test_round_trips_through_heading_in_image(self, platform):
        headings = numpy.array([0.0, 45.0, 90.0, 181.0, 359.0])
        image = heading_in_image(headings, platform)
        assert heading_from_image(image, platform) == pytest.approx(headings)

    def test_image_up_points_against_the_platform_heading(self):
        # image "up" (decreasing row) is the platform heading + 180
        assert heading_from_image(0.0, 10.0) == pytest.approx(190.0)

    def test_result_in_range(self):
        out = heading_from_image(numpy.linspace(-720, 720, 97), 345.0)
        assert ((out >= 0) & (out < 360)).all()


class TestAzimuthShiftM:
    platform_heading = -168.038694
    incidence = 35.0
    slant_range = 900_000.0
    v_sat = 7400.0  # Sentinel-1's orbital speed, m/s

    def test_along_track_motion_is_zero(self):
        look_direction = (self.platform_heading + 90.0) % 360.0
        along_track = (look_direction + 90.0) % 360.0
        got = azimuth_shift_m(
            10.0, along_track, self.platform_heading, self.incidence,
            self.slant_range, self.v_sat,
        )
        assert got == pytest.approx(0.0, abs=1e-9)

    def test_moving_away_gives_negative_shift(self):
        # receding target -> earlier azimuth time -> lower row index
        look_direction = (self.platform_heading + 90.0) % 360.0
        got = azimuth_shift_m(
            10.0, look_direction, self.platform_heading, self.incidence,
            self.slant_range, self.v_sat,
        )
        assert got < 0

    def test_moving_toward_gives_positive_shift(self):
        # approaching target -> later azimuth time -> higher row index
        look_direction = (self.platform_heading + 90.0) % 360.0
        toward = (look_direction + 180.0) % 360.0
        got = azimuth_shift_m(
            10.0, toward, self.platform_heading, self.incidence,
            self.slant_range, self.v_sat,
        )
        assert got > 0

    def test_magnitude_matches_closed_form(self):
        look_direction = (self.platform_heading + 90.0) % 360.0
        speed = 12.0
        got = azimuth_shift_m(
            speed, look_direction, self.platform_heading, self.incidence,
            self.slant_range, self.v_sat,
        )
        expected = -(
            speed * numpy.sin(numpy.deg2rad(self.incidence))
            * self.slant_range / self.v_sat
        )
        assert got == pytest.approx(expected)


# -- the orbit-based, ground-frame moving-target shift ---------------------------
class TestEllipsoid:
    def test_known_ecef(self):
        # on the equator / prime meridian, and at the pole
        assert sar.geodetic_to_ecef(0.0, 0.0)[0] == pytest.approx([sar.WGS84_A, 0.0, 0.0])
        b = sar.WGS84_A * numpy.sqrt(1 - sar.WGS84_E2)
        assert sar.geodetic_to_ecef(90.0, 0.0)[0] == pytest.approx([0.0, 0.0, b], abs=1e-6)

    def test_enu_axes_are_orthonormal_and_up_is_the_normal(self):
        axes = sar.enu_axes([10.0, -60.0], [20.0, 170.0])
        for a in axes:
            assert a @ a.T == pytest.approx(numpy.eye(3), abs=1e-12)
        # a step up the normal leaves lat / lon unchanged
        step = sar.geodetic_to_ecef(-60.0, 170.0, 100.0)[0] - sar.geodetic_to_ecef(-60.0, 170.0)[0]
        assert axes[1] @ step == pytest.approx([0.0, 0.0, 100.0], abs=1e-6)

    def test_offset_latlon_matches_ecef_distance(self):
        moved = sar.offset_latlon(55.0, 10.0, 1000.0, 30.0)[0]
        d = sar.geodetic_to_ecef(*moved)[0] - sar.geodetic_to_ecef(55.0, 10.0)[0]
        east, north, _ = sar.enu_axes(55.0, 10.0)[0] @ d
        assert numpy.hypot(east, north) == pytest.approx(1000.0, rel=1e-5)
        # a constant-bearing step, not the chord: off by half the meridians'
        # convergence over it (~0.003 degrees, 5 cm here)
        assert numpy.rad2deg(numpy.arctan2(east, north)) == pytest.approx(30.0, abs=0.01)


def _circular_orbit(radius=7_000e3, n=21, dt=1.0):
    """An equatorial circular orbit, analytic position and velocity."""
    omega = numpy.sqrt(sar.EARTH_GM / radius**3)
    t = (numpy.arange(n) - n // 2) * dt
    pos = radius * numpy.column_stack((numpy.cos(omega * t), numpy.sin(omega * t), numpy.zeros(n)))
    vel = radius * omega * numpy.column_stack((-numpy.sin(omega * t), numpy.cos(omega * t), numpy.zeros(n)))
    return t, pos, vel, omega


class TestOrbit:
    def test_hermite_reproduces_a_circular_orbit(self):
        t, pos, vel, omega = _circular_orbit(dt=2.0)
        got_p, got_v = sar.interpolate_orbit(t, pos, vel, [0.3, -4.7, 9.99])
        for tt, p, v in zip([0.3, -4.7, 9.99], got_p, got_v):
            r = 7_000e3
            assert p == pytest.approx([r * numpy.cos(omega * tt), r * numpy.sin(omega * tt), 0], abs=1e-3)
            assert v == pytest.approx(r * omega * numpy.array([-numpy.sin(omega * tt), numpy.cos(omega * tt), 0]), abs=1e-5)

    def test_imaging_time_is_zero_doppler_and_honours_the_cone(self):
        t, pos, vel, omega = _circular_orbit()
        # a target on the equator, 0.3 degrees of longitude ahead: overflown at
        # 0.3 degrees of orbit
        target = sar.geodetic_to_ecef(0.0, 0.3)
        t0 = sar.imaging_time(t, pos, vel, target)[0]
        assert t0 == pytest.approx(numpy.deg2rad(0.3) / omega, rel=1e-6)
        p, v = sar.interpolate_orbit(t, pos, vel, t0)
        assert sar.doppler_cone_cosine(p, v, target)[0] == pytest.approx(0.0, abs=1e-9)
        # seen earlier, still closing, at a positive cone cosine
        t1 = sar.imaging_time(t, pos, vel, target, 0.05)[0]
        p, v = sar.interpolate_orbit(t, pos, vel, t1)
        assert t1 < t0
        assert sar.doppler_cone_cosine(p, v, target)[0] == pytest.approx(0.05, abs=1e-9)


class TestMovingTargetShift:
    lat, lon = 40.0, 5.0
    incidence, slant_range, speed_s = 35.0, 650e3, 7_500.0

    def _sensor(self, look_bearing, squint_deg=0.0, right_looking=True):
        """Sensor state imaging (lat, lon) looking ``look_bearing``, its
        velocity horizontal and turned ``squint_deg`` off square to the LOS."""
        axes = sar.enu_axes(self.lat, self.lon)[0]
        inc, back = numpy.deg2rad(self.incidence), numpy.deg2rad(look_bearing + 180)
        u = numpy.array([numpy.sin(inc) * numpy.sin(back), numpy.sin(inc) * numpy.cos(back), numpy.cos(inc)])
        position = sar.geodetic_to_ecef(self.lat, self.lon)[0] + self.slant_range * axes.T @ u
        flight = numpy.deg2rad((look_bearing - 90 if right_looking else look_bearing + 90) + squint_deg)
        velocity = self.speed_s * axes.T @ numpy.array([numpy.sin(flight), numpy.cos(flight), 0])
        return position, velocity

    def test_matches_the_broadside_textbook_shift(self):
        look = 100.0
        position, velocity = self._sensor(look)
        for heading in (0.0, 45.0, 170.0, 280.0, 333.0):
            shift, bearing = sar.moving_target_shift(self.lat, self.lon, position, velocity, 8.0, heading)
            platform_heading = look - 90
            expected = azimuth_shift_m(8.0, heading, platform_heading, self.incidence, self.slant_range, self.speed_s)
            assert shift[0] == pytest.approx(expected, rel=1e-9, abs=1e-9)
            assert bearing[0] == pytest.approx(platform_heading % 360)

    def test_left_looking_flips_the_bearing_not_the_physics(self):
        look = 100.0
        towards = look + 180
        right = sar.moving_target_shift(self.lat, self.lon, *self._sensor(look), 8.0, towards)
        left = sar.moving_target_shift(self.lat, self.lon, *self._sensor(look, right_looking=False), 8.0, towards)
        assert left[0][0] == pytest.approx(right[0][0]) and left[0][0] > 0
        assert (left[1][0] - right[1][0]) % 360 == pytest.approx(180.0)

    def test_squint_divides_by_the_velocity_square_to_the_los(self):
        look, towards = 100.0, 280.0
        broadside, _ = sar.moving_target_shift(self.lat, self.lon, *self._sensor(look), 8.0, towards)
        squinted, _ = sar.moving_target_shift(self.lat, self.lon, *self._sensor(look, 30.0), 8.0, towards)
        assert squinted[0] == pytest.approx(broadside[0] / numpy.cos(numpy.deg2rad(30.0)))

    def test_look_bearing(self):
        position, _ = self._sensor(100.0)
        assert sar.look_bearing(self.lat, self.lon, position)[0] == pytest.approx(100.0)
