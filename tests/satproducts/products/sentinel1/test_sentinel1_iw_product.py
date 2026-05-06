import os
import time
import unittest

import numpy

from satproducts.common.image_slice import ImageSlice
from satproducts.products.sentinel.sentinel1.iw.sentinel1_iw_product import Sentinel1IWProduct
from tests.test_helper import BASE_DIR


class Sentinel1IWProductTest(unittest.TestCase):
    PRODUCT_PATH = os.path.join(BASE_DIR, 'data', 'products_minified',
                                'S1A_IW_GRDH_1SDV_20210102T224803_20210102T224828_035965_04368F_4953_COG.SAFE')
    PRODUCT = Sentinel1IWProduct

    def test_read_slice_complete_slice(self):
        # Arrange
        product = self.PRODUCT(self.PRODUCT_PATH)
        image_slice = ImageSlice(0, 0, 100, 100)

        # Act
        subset = product.read_slice(image_slice)

        # Assert
        self.assertEqual(subset.shape, (product.channels, 100, 100))
        self.assertEqual(subset.attrs["i"], image_slice.i)
        self.assertEqual(subset.attrs["j"], image_slice.j)

    def test_read_slice_incomplete_slice(self):
        # Arrange
        product = self.PRODUCT(self.PRODUCT_PATH)
        image_slice = ImageSlice(product.width - 50, product.height - 50, 100, 100)

        # Act
        subset = product.read_slice(image_slice)

        # Assert
        self.assertEqual(subset.shape, (product.channels, 50, 50))
        self.assertEqual(subset.attrs["i"], image_slice.i)
        self.assertEqual(subset.attrs["j"], image_slice.j)

    def test_read_slice_empty_slice(self):
        # Arrange
        product = self.PRODUCT(self.PRODUCT_PATH)
        image_slice = ImageSlice(product.width + 10, product.height + 10, 100, 100)

        # Act
        subset = product.read_slice(image_slice)

        # Assert
        self.assertEqual(subset.shape, (product.channels, 0, 0))
        self.assertEqual(subset.attrs["i"], image_slice.i)
        self.assertEqual(subset.attrs["j"], image_slice.j)

    def test_view_slice(self):
        # Arrange
        product = self.PRODUCT(self.PRODUCT_PATH)
        image_slice = ImageSlice(0, 0, 512, 512)

        # Act
        image = product.view(image_slice=image_slice)

        # Assert
        self.assertEqual(image.width, 512)
        self.assertEqual(image.height, 512)

    def test_get_transformer(self):
        # Arrange
        product = self.PRODUCT(self.PRODUCT_PATH)

        # Act
        transformer = product.transformer

        # Assert
        self.assertFalse(transformer is None)

    def test_identity_transform(self):
        # Arrange
        product = self.PRODUCT(self.PRODUCT_PATH)
        rowcol1 = numpy.asarray([[10, 10]])

        # Act
        transformer = product.transformer
        latlon1 = transformer.rowcol_to_latlon(rowcol1)
        rowcol2 = transformer.latlon_to_rowcol(latlon1)
        latlon2 = transformer.rowcol_to_latlon(rowcol2)

        # Arrange
        self.assertFalse(transformer is None)
        self.assertTrue(numpy.allclose(latlon1, latlon2, atol=0.001))
        self.assertTrue(numpy.allclose(rowcol1, rowcol2, atol=3))

    def test_footprint_and_timestamp(self):
        # Arrange + Act
        product = self.PRODUCT(self.PRODUCT_PATH)

        # Assert
        self.assertTrue(product.timestamp is not None)
        self.assertTrue(product.footprint is not None)

    def test_tile(self):
        product = self.PRODUCT(self.PRODUCT_PATH)
        for img_slice, tile in product.tile(512):
            print(img_slice)
            print(tile.shape)
            break
