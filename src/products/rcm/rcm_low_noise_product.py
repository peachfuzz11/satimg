from products.rcm.rcm_product import RCMProduct


class RCM50mLowNoiseProduct(RCMProduct):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _normalize_image(self, x):
        # Remove NaNs or Inf values if present
        # x = numpy.nan_to_num(x, nan=0.0, posinf=50010.0, neginf=0.0)

        # Clip and normalize
        return (x / x.max()) * 255.0

    def validate(self):
        self._valid = True
        # Add validation logic specific to 50m low-noise product
        if not self._product_path:
            self._valid = False
            raise ValueError("Product path is missing.")

        # Perform other validation checks as necessary
