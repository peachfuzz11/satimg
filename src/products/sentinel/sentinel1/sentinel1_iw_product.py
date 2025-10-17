from products.sentinel.sentinel1.sentinel1_product import Sentinel1Product


class Sentinel1IWProduct(Sentinel1Product):
    def _normalize_fn(self, x):
        return x.clip(0, 510.0) * 255.0 / 510.0
