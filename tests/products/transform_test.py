import abc

import numpy


class TransformTest(abc.ABC):
    PRODUCT_PATH = None
    PRODUCT = None

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
