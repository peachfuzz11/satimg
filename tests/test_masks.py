import numpy
import pytest

from satimg.fishfarms import FishFarmMask
from satimg.landmask import LandMask
from satimg.platforms import PlatformMask
from satimg.ports import HarborMask, PortMask, features_in_bbox
from satimg.windfarms import WindfarmMask

# (lat, lon)
COPENHAGEN_CITY_HALL = (55.6757, 12.5694)
SAINT_NAZAIRE = (47.1557, -2.6009)  # wind farm built 2022
VESTERHAV_SYD = (56.1041, 7.961)  # wind farm without a build year
CHINA_TURBINES = (33.16136, 120.80102)  # offshore Jiangsu turbine OSM maps as a point only
JOHAN_SVERDRUP = (58.8361, 2.54862)  # production platform
BJOROYA_SALMON = (64.45463, 10.63072)  # EMODnet salmon farm
CHILE_SALMON = (-45.39506, -72.8679)  # OSM salmon farm
PORT_OF_LOS_ANGELES = (33.74456, -118.26165)
MARINA_DEL_REY = (33.97245, -118.45101)
BILBAO_PORT = (43.34676, -3.06383)  # untagged in OSM, 20 ha or more
NYHAVN = (55.67972, 12.59092)  # untagged in OSM, no port next to it
OPEN_SEA = (45.0, -10.0)


def contains(mask, *points):
    return mask.contains([p[0] for p in points], [p[1] for p in points]).tolist()


@pytest.mark.parametrize("mask", [LandMask(), WindfarmMask(), PortMask(), HarborMask(), PlatformMask(), FishFarmMask()],
                         ids=lambda m: type(m).__name__)
class TestEveryMask:
    def test_a_scalar_point(self, mask):
        assert mask.contains(*OPEN_SEA).shape == ()

    def test_keeps_the_shape(self, mask):
        assert mask.contains(numpy.full((2, 3), 45.0), numpy.full((2, 3), -10.0)).shape == (2, 3)

    def test_no_points(self, mask):
        assert contains(mask) == []


def test_land():
    assert contains(LandMask(), COPENHAGEN_CITY_HALL, OPEN_SEA) == [True, False]


class TestWindfarmMask:
    def test_points_inside_built_farms(self):
        assert contains(WindfarmMask(year=2026), SAINT_NAZAIRE, OPEN_SEA, VESTERHAV_SYD) == [True, False, True]

    def test_farm_built_after_the_year_is_ignored(self):
        assert contains(WindfarmMask(year=2021), SAINT_NAZAIRE, VESTERHAV_SYD) == [False, True]

    def test_turbines_mapped_as_points_count_whatever_the_year(self):
        assert contains(WindfarmMask(year=2000), CHINA_TURBINES) == [True]

    def test_no_year_matches_every_farm(self):
        assert contains(WindfarmMask(), SAINT_NAZAIRE) == [True]


class TestPortsAndHarbors:
    def test_ports(self):
        assert contains(PortMask(), PORT_OF_LOS_ANGELES, MARINA_DEL_REY, OPEN_SEA) == [True, False, False]

    def test_harbors(self):
        assert contains(HarborMask(), PORT_OF_LOS_ANGELES, MARINA_DEL_REY, OPEN_SEA) == [False, True, False]

    def test_untagged_areas_take_their_kind_from_size_and_neighbours(self):
        assert contains(PortMask(), BILBAO_PORT, NYHAVN) == [True, False]
        assert contains(HarborMask(), BILBAO_PORT, NYHAVN) == [False, True]

    def test_features_in_bbox(self):
        names = [f["properties"]["name"] for f in features_in_bbox(-118.5, 33.7, -118.2, 34.0)]
        assert "Port of Los Angeles" in names
        assert "Marina del Rey" in names

    def test_open_sea_has_no_features(self):
        assert features_in_bbox(-11, 44, -10, 45) == []


def test_platforms():
    assert contains(PlatformMask(), JOHAN_SVERDRUP, OPEN_SEA) == [True, False]


def test_fish_farms():
    assert contains(FishFarmMask(), BJOROYA_SALMON, CHILE_SALMON, OPEN_SEA) == [True, True, False]
