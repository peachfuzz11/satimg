"""EO/optical-specific per-pixel geometry: see ``satimg/eo_utils.py``."""

import numpy
import pytest

from satimg.eo_utils import heading_in_image


class TestHeadingInImage:
    def test_identity_for_in_range_heading(self):
        for heading in numpy.linspace(0, 360, 9, endpoint=False):
            assert heading_in_image(heading) == pytest.approx(heading % 360.0)

    def test_wraps_negative_and_over_360(self):
        assert heading_in_image(-10.0) == pytest.approx(350.0)
        assert heading_in_image(370.0) == pytest.approx(10.0)

    def test_vectorised(self):
        got = heading_in_image(numpy.array([0.0, 90.0, -10.0, 370.0]))
        numpy.testing.assert_allclose(got, [0.0, 90.0, 350.0, 10.0])
