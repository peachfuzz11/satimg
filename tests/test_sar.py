"""SAR geometry: the ellipsoid / orbit helpers and the ground-frame
moving-target shift. See ``satimg/sar.py`` for the derivation and sign
conventions."""

import numpy
import pytest

from satimg import sar
from satimg.sar import ground_track_heading


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
        platform_heading = look - 90
        for heading in (0.0, 45.0, 170.0, 280.0, 333.0):
            shift, bearing = sar.moving_target_shift(self.lat, self.lon, position, velocity, 8.0, heading)
            # textbook: -v_r R / V, v_r the range rate (receding positive)
            receding = 8.0 * numpy.cos(numpy.deg2rad(heading - look)) * numpy.sin(numpy.deg2rad(self.incidence))
            expected = -receding * self.slant_range / self.speed_s
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
