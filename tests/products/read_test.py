import abc

from satproducts.common.image_slice import ImageSlice


class ReadTest(abc.ABC):
    PRODUCT_PATH = None
    PRODUCT = None

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

    def test_view_full(self):
        # Arrange
        product = self.PRODUCT(self.PRODUCT_PATH)

        # Act
        image = product.view()

        # Assert
        self.assertEqual(image.width, product.width)
        self.assertEqual(image.height, product.height)

    def test_view_slice(self):
        # Arrange
        product = self.PRODUCT(self.PRODUCT_PATH)
        image_slice = ImageSlice(0, 0, 100, 100)

        # Act
        image = product.view(image_slice)

        # Assert
        self.assertEqual(image.width, 100)
        self.assertEqual(image.height, 100)
