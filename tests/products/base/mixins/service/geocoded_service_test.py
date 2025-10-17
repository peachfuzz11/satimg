import unittest

from products.base.mixins.service.geocoded_service import get_bounds_from_footprint


class GeocodedServiceTest(unittest.TestCase):

    def test_get_bounds_from_footprint(self):
        # Arrange
        test_footprint = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [(13.579708268630132, 55.280642797207044),
                                (13.53512454747758, 55.20776443652352),
                                (13.448200901895712, 55.06504300021859),
                                (13.376784334281517, 54.946938457754705),
                                (11.879220117042458, 54.91902653885059),
                                (11.800511438780461, 55.90416762514414),
                                (13.55602214889276, 55.93727334575647),
                                (13.579708268630132, 55.280642797207044)]
            }
        }
        test_max_lon = 13.579708268630132
        test_min_lon = 11.800511438780461
        test_max_lat = 55.93727334575647
        test_min_lat = 54.91902653885059

        # Act
        max_lon, min_lon, max_lat, min_lat = get_bounds_from_footprint(test_footprint)

        # Assert
        self.assertEqual(max_lon, test_max_lon)
        self.assertEqual(min_lon, test_min_lon)
        self.assertEqual(max_lat, test_max_lat)
        self.assertEqual(min_lat, test_min_lat)
