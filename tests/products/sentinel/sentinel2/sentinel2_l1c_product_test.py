import os
import unittest

from common.image_slice import ImageSlice
from products.sentinel.sentinel2.sentinel2_l1c_product import Sentinel2L1CProduct
from tests.test_helper import BASE_DIR


class Sentinel2L1CProductTest(unittest.TestCase):
    file_path = os.path.join(BASE_DIR, 'data', 'products_minified',
                             'S2A_MSIL1C_20220114T103401_N0301_R108_T33UUB_20220114T123457.SAFE')

    def test_read_slice_complete_slice(self):
        # Arrange
        product = Sentinel2L1CProduct(self.file_path)
        image_slice = ImageSlice(0, 0, 100, 100)

        # Act
        with product:
            subset = product.read_slice(image_slice)

        # Assert
        self.assertEqual(subset.shape, (13, 100, 100))
        self.assertEqual(subset.attrs["i"], image_slice.i)
        self.assertEqual(subset.attrs["j"], image_slice.j)

    def test_read_slice_incomplete_slice(self):
        # Arrange
        product = Sentinel2L1CProduct(self.file_path)
        image_slice = ImageSlice(10900, 10900, 100, 100)

        # Act
        with product:
            subset = product.read_slice(image_slice)

        # Assert
        self.assertEqual(subset.shape, (13, 80, 80))
        self.assertEqual(subset.attrs["i"], image_slice.i)
        self.assertEqual(subset.attrs["j"], image_slice.j)

    def test_read_slice_empty_slice(self):
        # Arrange
        product = Sentinel2L1CProduct(self.file_path)
        image_slice = ImageSlice(20000, 20000, 100, 100)

        # Act
        with product:
            subset = product.read_slice(image_slice)

        # Assert
        self.assertEqual(subset.shape, (13, 0, 0))
        self.assertEqual(subset.attrs["i"], image_slice.i)
        self.assertEqual(subset.attrs["j"], image_slice.j)

    def test_view_full(self):
        # Arrange
        product = Sentinel2L1CProduct(self.file_path)

        # Act
        with product:
            image = product.view_full()

        # Assert
        self.assertEqual(image.width, 10980)
        self.assertEqual(image.height, 10980)
        self.assertEqual(len(image.getbands()), 3)

    def test_view_slice(self):
        # Arrange
        product = Sentinel2L1CProduct(self.file_path)
        image_slice = ImageSlice(5000, 5000, 100, 100)

        # Act
        with product:
            image = product.view_slice(image_slice)

        # Assert
        self.assertEqual(image.width, 100)
        self.assertEqual(image.height, 100)
        self.assertEqual(len(image.getbands()), 3)
