import PIL
import numpy

from satproducts.prediction.models.model import Model


class Yolo26Model(Model):

    def predict(self, subset: PIL.Image, *args, **kwargs):
        subset = subset if subset.mode == "RGB" else subset.convert("RGB")
        subset = numpy.einsum("ijk->kij", numpy.asarray(subset))
        subset = subset.astype(numpy.float32)
        subset = pad_axis_to_size(subset, axis=1, target_size=256)
        subset = pad_axis_to_size(subset, axis=2, target_size=256)

        subset = numpy.expand_dims(subset, axis=0)
        detections = self._infer(subset, *args, **kwargs)
        detections = detections[..., 0:5]
        detections = numpy.squeeze(detections, axis=0)[:, 0:5]

        conf_threshold = kwargs.get("conf_threshold", 0.2)
        detections = detections[(detections[:, 4] > conf_threshold)]
        return detections


def pad_axis_to_size(array: numpy.ndarray, axis: int, target_size: int) -> numpy.ndarray:
    current_size = array.shape[axis]
    if current_size >= target_size:
        return array
    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (0, target_size - current_size)
    return numpy.pad(array, pad_width=pad_width, mode='constant', constant_values=0)
