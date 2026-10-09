"""SAR-specific per-pixel geometry: heading conversion and the moving-target
azimuth-shift displacement itself. See ``satimg/sar_utils.py`` for the
derivation and sign conventions."""

import numpy
import pytest

from satimg.sar_utils import (
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
