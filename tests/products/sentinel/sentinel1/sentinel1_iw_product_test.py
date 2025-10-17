import os
import unittest

from tests.test_helper import BASE_DIR
from common.image_slice import ImageSlice
from products.sentinel.sentinel1.sentinel1_iw_product import Sentinel1IWProduct


class Sentinel1IWProductTest(unittest.TestCase):
    file_path = os.path.join(BASE_DIR, 'data', 'products_minified',
                             'S1A_IW_GRDH_1SDV_20210102T224803_20210102T224828_035965_04368F_4953_COG.SAFE')

    def test_read_slice_complete_slice(self):
        # Arrange
        product = Sentinel1IWProduct(self.file_path)
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
        product = Sentinel1IWProduct(self.file_path)
        image_slice = ImageSlice(25300, 16800, 100, 100)

        # Act
        with product:
            subset = product.read_slice(image_slice)

        # Assert
        self.assertEqual(subset.shape, (2, 15, 49))
        self.assertEqual(subset.attrs["i"], image_slice.i)
        self.assertEqual(subset.attrs["j"], image_slice.j)

    def test_read_slice_empty_slice(self):
        # Arrange
        product = Sentinel1IWProduct(self.file_path)
        image_slice = ImageSlice(26000, 17000, 100, 100)

        # Act
        with product:
            subset = product.read_slice(image_slice)

        # Assert
        self.assertEqual(subset.shape, (2, 0, 0))
        self.assertEqual(subset.attrs["i"], image_slice.i)
        self.assertEqual(subset.attrs["j"], image_slice.j)

    def test_view_full(self):
        # Arrange
        product = Sentinel1IWProduct(self.file_path)

        # Act
        with product:
            image = product.view_full()

        # Assert
        self.assertEqual(image.height, 16815)
        self.assertEqual(image.width, 25349)
        self.assertEqual(len(image.getbands()), 1)

    def test_view_slice(self):
        # Arrange
        product = Sentinel1IWProduct(self.file_path)
        image_slice = ImageSlice(5000, 5000, 100, 100)

        # Act
        with product:
            image = product.view_slice(image_slice)

        # Assert
        self.assertEqual(image.width, 100)
        self.assertEqual(image.height, 100)
        self.assertEqual(len(image.getbands()), 1)

    def test_footprint_and_timestamp(self):
        # Arrange + Act
        product = Sentinel1IWProduct(self.file_path)

        # Assert
        assert product.timestamp is not None
        assert product.footprint is not None
        assert product.width is not None
        assert product.height is not None
