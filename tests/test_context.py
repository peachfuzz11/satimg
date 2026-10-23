import datetime
from types import SimpleNamespace

import numpy

from satimg.prediction import Detector
from satimg.prediction.context import FISHERY, IN_HARBOR, IN_PORT, ON_LAND, PLATFORM, TURBINE, context_tags

# (lat, lon)
COPENHAGEN_CITY_HALL = (55.6757, 12.5694)
SAINT_NAZAIRE = (47.1557, -2.6009)  # wind farm built 2022
JOHAN_SVERDRUP = (58.8361, 2.54862)  # production platform
BJOROYA_SALMON = (64.45463, 10.63072)  # salmon farm
PORT_OF_LOS_ANGELES = (33.74456, -118.26165)
MARINA_DEL_REY = (33.97245, -118.45101)
OPEN_SEA = (45.0, -10.0)


def tags(*points, year=None):
    return context_tags([p[0] for p in points], [p[1] for p in points], year=year)


class TestContextTags:
    def test_tags_where_each_point_lies(self):
        assert tags(COPENHAGEN_CITY_HALL, SAINT_NAZAIRE, JOHAN_SVERDRUP, BJOROYA_SALMON, OPEN_SEA) == [
            [ON_LAND], [TURBINE], [PLATFORM], [FISHERY], []]

    def test_ports_and_harbors(self):
        # the port point is on terminal land
        assert tags(PORT_OF_LOS_ANGELES, MARINA_DEL_REY) == [[ON_LAND, IN_PORT], [IN_HARBOR]]

    def test_wind_farm_built_after_the_year_is_no_turbine(self):
        assert tags(SAINT_NAZAIRE, year=2021) == [[]]

    def test_no_points(self):
        assert tags() == []


class FakeTransformer:
    """Pixel (row, col) i maps to the i-th point."""

    def __init__(self, points):
        self._points = points

    def rowcol_to_latlon(self, rowcol):
        return numpy.array([self._points[int(row)] for row, _ in rowcol])


def detector(points, year=2026):
    product = SimpleNamespace(transformer=FakeTransformer(points),
                              timestamp=datetime.datetime(year, 6, 1, tzinfo=datetime.timezone.utc))
    return Detector(product, model=None)


def boxes(n):
    # box i is centred on pixel (row i, col 0)
    return numpy.array([[-1, i - 1, 1, i + 1, 0.9] for i in range(n)], dtype=float)


class TestDetectorTags:
    def test_detections_are_tagged_and_none_dropped(self):
        d = detector([OPEN_SEA, COPENHAGEN_CITY_HALL, SAINT_NAZAIRE])
        detections = d._to_detections(boxes(3), d._config)
        assert [x.tags for x in detections] == [[], [ON_LAND], [TURBINE]]

    def test_the_product_year_decides_the_wind_farms(self):
        d = detector([SAINT_NAZAIRE], year=2021)
        assert [x.tags for x in d._to_detections(boxes(1), d._config)] == [[]]

    def test_no_boxes(self):
        d = detector([])
        assert d._to_detections(numpy.zeros((0, 5)), d._config) == []
