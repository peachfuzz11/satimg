import os
import unittest

import numpy

from products.sentinel.sentinel1.sentinel1_iw_product import Sentinel1IWProduct
from sat_product_factory import SatProductFactory
from tests.test_helper import BASE_DIR


class TransformS1IWTest(unittest.TestCase):
    S1_IW_PRODUCT_PATH = os.path.join(BASE_DIR, 'data', 'products_minified',
                                      'S1A_IW_GRDH_1SDH_20141006T074149_20141006T074214_002706_00306E_51E0_COG.SAFE')

    def test_get_transformer(self):
        # Arrange
        product: Sentinel1IWProduct = SatProductFactory(self.S1_IW_PRODUCT_PATH).create()

        # Act
        transformer = product.get_transformer()

        # Assert
        self.assertFalse(transformer is None)

    def test_identity_transform(self):
        # Arrange
        product: Sentinel1IWProduct = SatProductFactory(self.S1_IW_PRODUCT_PATH).create()
        latlon1 = numpy.asarray([[63, -17]])

        # Act
        transformer = product.get_transformer()
        rowcol1 = transformer.latlon_to_rowcol(latlon1)
        latlon2 = transformer.rowcol_to_latlon(rowcol1)
        rowcol2 = transformer.latlon_to_rowcol(latlon2)

        # Arrange
        self.assertFalse(transformer is None)
        self.assertTrue(numpy.allclose(latlon1, latlon2, atol=0.001))
        self.assertTrue(numpy.allclose(rowcol1, rowcol2, atol=3))
