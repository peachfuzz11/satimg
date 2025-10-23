import abc
import os
import unittest

import numpy

from satproducts.common.image_slice import ImageSlice
from tests.test_helper import BASE_DIR


class ProductTest(abc.ABC, unittest.TestCase):
    MINI_PATH = os.path.join(BASE_DIR, 'data', 'products_minified')
    PRODUCT_PATH = None
    PRODUCT = None

    def skip(self):
        if self.PRODUCT_PATH is None or self.PRODUCT is None:
            self.skipTest("NONE")

    def test_read_slice_complete_slice(self):
        self.skip()
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
        self.skip()
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
        self.skip()
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
        self.skip()
        # Arrange
        product = self.PRODUCT(self.PRODUCT_PATH)
        image_slice = ImageSlice(0, 0, 100, 100)

        # Act
        image = product.view(image_slice=image_slice)

        # Assert
        self.assertEqual(image.width, 100)
        self.assertEqual(image.height, 100)

    def test_get_transformer(self):
        self.skip()
        # Arrange
        product = self.PRODUCT(self.PRODUCT_PATH)

        # Act
        transformer = product.transformer

        # Assert
        self.assertFalse(transformer is None)

    def test_identity_transform(self):
        self.skip()
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
        self.skip()
        # Arrange + Act
        product = self.PRODUCT(self.PRODUCT_PATH)

        # Assert
        self.assertTrue(product.timestamp is not None)
        self.assertTrue(product.footprint is not None)
