import numpy

from products.rcm.rcm_product import RCMProduct


class RCM50mProduct(RCMProduct):
    def _normalize_image(self, x):
        # Remove NaNs or Inf values if present
        x = numpy.nan_to_num(x, nan=0.0, posinf=5010.0, neginf=0.0)

        # Clip and normalize
        return numpy.clip(x, 0, 5010.0) * 255.0 / 5010.0

    def validate(self):
        self._valid = True
        