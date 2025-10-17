import os
import unittest

from common.image_slice import ImageSlice
from products.sentinel.sentinel1.sentinel1_ew_product import Sentinel1EWProduct
from tests.test_helper import BASE_DIR


class Sentinel1EWProductTest(unittest.TestCase):
    file_path = os.path.join(BASE_DIR, 'data', 'products_minified',
                             'S1A_EW_GRDM_1SDH_20240215T195548_20240215T195633_052574_065C0C_17E5_COG.SAFE')

    def test_read_slice_complete_slice(self):
        # Arrange
        product = Sentinel1EWProduct(self.file_path)
        image_slice = ImageSlice(0, 0, 100, 100)

        # Act
        with product:
            subset = product.read_slice(image_slice)

        # Assert
        self.assertEqual(subset.shape, (2, 100, 100))
        self.assertEqual(subset.attrs["i"], image_slice.i)
        self.assertEqual(subset.attrs["j"], image_slice.j)

    def test_read_slice_incomplete_slice(self):
        # Arrange
        product = Sentinel1EWProduct(self.file_path)
        image_slice = ImageSlice(10400, 7400, 100, 100)

        # Act
        with product:
            subset = product.read_slice(image_slice)

        # Assert
        self.assertEqual(subset.shape, (2, 67, 12))
        self.assertEqual(subset.attrs["i"], image_slice.i)
        self.assertEqual(subset.attrs["j"], image_slice.j)

    def test_read_slice_empty_slice(self):
        # Arrange
        product = Sentinel1EWProduct(self.file_path)
        image_slice = ImageSlice(11000, 8000, 100, 100)

        # Act
        with product:
            subset = product.read_slice(image_slice)

        # Assert
        self.assertEqual(subset.shape, (2, 0, 0))
        self.assertEqual(subset.attrs["i"], image_slice.i)
        self.assertEqual(subset.attrs["j"], image_slice.j)

    def test_view_full(self):
        # Arrange
        product = Sentinel1EWProduct(self.file_path)

        # Act
        with product:
            image = product.view_full()

        # Assert
        self.assertEqual(image.height, 7467)
        self.assertEqual(image.width, 10412)
        self.assertEqual(len(image.getbands()), 1)

    def test_view_slice(self):
        # Arrange
        product = Sentinel1EWProduct(self.file_path)
        image_slice = ImageSlice(5000, 5000, 100, 100)

        # Act
        with product:
            image = product.view_slice(image_slice)

        # Assert
        self.assertEqual(image.width, 100)
        self.assertEqual(image.height, 100)
        self.assertEqual(len(image.getbands()), 1)