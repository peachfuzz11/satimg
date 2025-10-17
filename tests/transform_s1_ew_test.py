import os
import unittest

import numpy

from products.sentinel.sentinel1.sentinel1_ew_product import Sentinel1EWProduct
from sat_product_factory import SatProductFactory
from tests.test_helper import BASE_DIR


class TransformS1IWTest(unittest.TestCase):
    S1_EW_PRODUCT_PATH = os.path.join(BASE_DIR, 'data', 'products_minified',
                                      'S1A_EW_GRDM_1SDH_20240215T195548_20240215T195633_052574_065C0C_17E5_COG.SAFE')

    def test_get_transformer(self):
        # Arrange
        product: Sentinel1EWProduct = SatProductFactory(self.S1_EW_PRODUCT_PATH).create()

        # Act
        transformer = product.get_transformer()

        # Assert
        self.assertFalse(transformer is None)

    def test_identity_transform(self):
        # Arrange
        product: Sentinel1EWProduct = SatProductFactory(self.S1_EW_PRODUCT_PATH).create()
        latlon1 = numpy.asarray([[62, -30]])

        # Act
        transformer = product.get_transformer()
        rowcol1 = transformer.latlon_to_rowcol(latlon1)
        latlon2 = transformer.rowcol_to_latlon(rowcol1)
        rowcol2 = transformer.latlon_to_rowcol(latlon2)

        # Arrange
        self.assertFalse(transformer is None)
        self.assertTrue(numpy.allclose(latlon1, latlon2, atol=0.01))
        self.assertTrue(numpy.allclose(rowcol1, rowcol2, atol=15))
