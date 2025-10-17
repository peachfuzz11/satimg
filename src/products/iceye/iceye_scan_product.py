from products.iceye.iceye_product import IceyeProduct


class IceyeScanProduct(IceyeProduct):
    def __init__(self, product_path):
        super().__init__(product_path)

    def validate(self):
        self._valid = True
        if not self._product_path:
            self._valid = False
            raise ValueError("Product path is missing.")
