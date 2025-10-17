from products.sentinel.sentinel1.sentinel1_product import Sentinel1Product


class Sentinel1EWProduct(Sentinel1Product):
    def _normalize_fn(self, x):
        return x.clip(0, 5010.0) * 255.0 / 5010.0
