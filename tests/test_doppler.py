"""SAR moving-target azimuth-shift geometry: heading conversion and the azimuth
displacement itself. See ``satimg/doppler.py`` for the derivation and sign
conventions."""

import numpy
import pytest

from satimg.doppler import azimuth_shift_m, heading_to_los


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

    def test_moving_away_gives_positive_shift(self):
        look_direction = (self.platform_heading + 90.0) % 360.0
        got = azimuth_shift_m(
            10.0, look_direction, self.platform_heading, self.incidence,
            self.slant_range, self.v_sat,
        )
        assert got > 0

    def test_moving_toward_gives_negative_shift(self):
        look_direction = (self.platform_heading + 90.0) % 360.0
        toward = (look_direction + 180.0) % 360.0
        got = azimuth_shift_m(
            10.0, toward, self.platform_heading, self.incidence,
            self.slant_range, self.v_sat,
        )
        assert got < 0

    def test_magnitude_matches_closed_form(self):
        look_direction = (self.platform_heading + 90.0) % 360.0
        speed = 12.0
        got = azimuth_shift_m(
            speed, look_direction, self.platform_heading, self.incidence,
            self.slant_range, self.v_sat,
        )
        expected = (
            speed * numpy.sin(numpy.deg2rad(self.incidence))
            * self.slant_range / self.v_sat
        )
        assert got == pytest.approx(expected)
