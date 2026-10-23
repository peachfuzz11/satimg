import numpy

from satimg.platforms import PlatformMask
from satimg.ports import HARBOR, PORT, PortMask
from satimg.windfarms import WindfarmMask

SAINT_NAZAIRE = (-2.6009, 47.1557)  # wind farm built 2022
VESTERHAV_SYD = (7.961, 56.1041)  # wind farm without a build year
CHINA_TURBINES = (120.80102, 33.16136)  # offshore Jiangsu turbine OSM maps as a point only
JOHAN_SVERDRUP = (2.54862, 58.8361)  # production platform
PORT_OF_LOS_ANGELES = (-118.26165, 33.74456)
MARINA_DEL_REY = (-118.45101, 33.97245)
BILBAO_PORT = (-3.06383, 43.34676)  # untagged in OSM, 20 ha or more
NYHAVN = (12.59092, 55.67972)  # untagged in OSM, no port next to it
OPEN_SEA = (-10.0, 45.0)


def lonlat(*points):
    return numpy.array([p[0] for p in points]), numpy.array([p[1] for p in points])


class TestWindfarmMask:
    def test_points_inside_built_farms(self):
        mask = WindfarmMask().contains_many(*lonlat(SAINT_NAZAIRE, OPEN_SEA, VESTERHAV_SYD), year=2026)
        assert mask.tolist() == [True, False, True]

    def test_farm_built_after_the_year_is_ignored(self):
        assert WindfarmMask().contains_many(*lonlat(SAINT_NAZAIRE, VESTERHAV_SYD), year=2021).tolist() == [False, True]

    def test_turbines_mapped_as_points_count_whatever_the_year(self):
        assert WindfarmMask().contains_many(*lonlat(CHINA_TURBINES), year=2000).tolist() == [True]

    def test_no_year_matches_every_farm(self):
        assert WindfarmMask().contains_many(*lonlat(SAINT_NAZAIRE)).tolist() == [True]

    def test_keeps_the_shape(self):
        assert WindfarmMask().contains_many(numpy.zeros((2, 3)), numpy.zeros((2, 3))).shape == (2, 3)

    def test_no_points(self):
        assert WindfarmMask().contains_many([], []).tolist() == []


class TestPortMask:
    def test_kinds(self):
        kinds = PortMask().kinds_many(*lonlat(PORT_OF_LOS_ANGELES, MARINA_DEL_REY, OPEN_SEA))
        assert kinds.tolist() == [PORT, HARBOR, None]

    def test_untagged_areas_take_their_kind_from_size_and_neighbours(self):
        assert PortMask().kinds_many(*lonlat(BILBAO_PORT, NYHAVN)).tolist() == [PORT, HARBOR]

    def test_no_points(self):
        assert PortMask().kinds_many([], []).tolist() == []

    def test_features_in_bbox(self):
        names = [f["properties"]["name"] for f in PortMask().features_in_bbox(-118.5, 33.7, -118.2, 34.0)]
        assert "Port of Los Angeles" in names
        assert "Marina del Rey" in names

    def test_open_sea_has_no_features(self):
        assert PortMask().features_in_bbox(-11, 44, -10, 45) == []


class TestPlatformMask:
    def test_points_at_a_platform(self):
        assert PlatformMask().contains_many(*lonlat(JOHAN_SVERDRUP, OPEN_SEA)).tolist() == [True, False]

    def test_no_points(self):
        assert PlatformMask().contains_many([], []).tolist() == []
