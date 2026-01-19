import numpy

from satproducts.prediction.models.model import Model


class DetectionModel(Model):

    def _infer(self, subset, *args, **kwargs):
        input_name = self.session.get_inputs()[0].name
        detections = self.session.run(None, {input_name: subset})[0]
        return detections

    def predict(self, subset, *args, **kwargs):
        subset = subset.astype(numpy.float32)
        subset = pad_axis_to_size(subset, axis=1, target_size=256)
        subset = pad_axis_to_size(subset, axis=2, target_size=256)

        subset = numpy.expand_dims(subset, axis=0)
        detections = self._infer(subset, *args, **kwargs)
        detections = numpy.einsum("ij->ji", numpy.squeeze(detections, axis=0))
        return detections


def pad_axis_to_size(array: numpy.ndarray, axis: int, target_size: int) -> numpy.ndarray:
    current_size = array.shape[axis]
    if current_size >= target_size:
        return array
    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (0, target_size - current_size)
    return numpy.pad(array, pad_width=pad_width, mode='constant', constant_values=0)
