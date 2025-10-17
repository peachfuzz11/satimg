import os
import unittest

import numpy
import rasterio

from products.sentinel.sentinel2.sentinel2_l1c_product import Sentinel2L1CProduct
from sat_product_factory import SatProductFactory
from tests.test_helper import BASE_DIR


class TransformS2L1CTests(unittest.TestCase):
    S2_L1C_PRODUCT_PATH = os.path.join(BASE_DIR, 'data', 'products_minified',
                                       'S2A_MSIL1C_20220114T103401_N0301_R108_T33UUB_20220114T123457.SAFE')

    def test_get_crs_and_transform(self):
        # Arrange
        product: Sentinel2L1CProduct = SatProductFactory(self.S2_L1C_PRODUCT_PATH).create()
        b2 = product._bands[1]

        # Act
        with rasterio.open(b2['file_path']) as src:
            crs = src.crs
            transform = src.transform

        # Assert
        self.assertFalse(crs is None)
        self.assertFalse(transform is None)

    def test_get_transformer(self):
        # Arrange
        product: Sentinel2L1CProduct = SatProductFactory(self.S2_L1C_PRODUCT_PATH).create()

        # Act
        transformer = product.get_transformer()

        # Assert
        self.assertFalse(transformer is None)

    def test_identity_transform(self):
        # Arrange
        product: Sentinel2L1CProduct = SatProductFactory(self.S2_L1C_PRODUCT_PATH).create()
        latlon1 = numpy.asarray([[55.686376758010525, 12.564620730957847]])

        # Act
        transformer = product.get_transformer()
        rowcol1 = transformer.latlon_to_rowcol(latlon1)
        latlon2 = transformer.rowcol_to_latlon(rowcol1)
        rowcol2 = transformer.latlon_to_rowcol(latlon2)

        # Arrange
        self.assertFalse(transformer is None)
        self.assertTrue(numpy.allclose(latlon1, latlon2))
        self.assertTrue(numpy.allclose(rowcol1, rowcol2))
