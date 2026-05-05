import PIL.Image
import numpy

from satproducts.prediction.models.model import Model


class Yolo26Model(Model):

    def predict(self, subset: PIL.Image, *args, **kwargs) -> numpy.ndarray:
        subset = subset if subset.mode == "RGB" else subset.convert("RGB")
        subset = numpy.einsum("ijk->kij", numpy.asarray(subset))
        subset = subset.astype(numpy.float32) / 255.0
        subset = pad_axis_to_size(subset, axis=1, target_size=kwargs.get("slice_size"))
        subset = pad_axis_to_size(subset, axis=2, target_size=kwargs.get("slice_size"))
        subset = numpy.expand_dims(subset, axis=0)
        detections = self._infer(subset, *args, **kwargs)
        detections = numpy.squeeze(detections, axis=0)
        return detections


def pad_axis_to_size(array: numpy.ndarray, axis: int, target_size: int) -> numpy.ndarray:
    current_size = array.shape[axis]
    if current_size >= target_size:
        return array
    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (0, target_size - current_size)
    return numpy.pad(array, pad_width=pad_width, mode='constant', constant_values=0)
