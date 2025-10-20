import abc


class HandlerTest(abc.ABC):
    PRODUCT_PATH = None
    PRODUCT = None
    HANDLER = None

    def test_handler_works(self):
        product = self.HANDLER().handle(product_path=self.PRODUCT_PATH)
        self.assertTrue(product is not None)
        self.assertTrue(isinstance(product, self.PRODUCT))

    def test_handler_fail(self):
        try:
            product = self.HANDLER().handle(product_path="")
            self.assertTrue(False)
        except Exception as e:
            self.assertTrue(True)
