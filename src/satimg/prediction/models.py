"""ONNX detection models. ``predict`` takes a ``(band, y, x)`` uint8 tile and
returns an ``(N, 5)`` array of ``x1, y1, x2, y2, confidence`` in tile pixels."""

from __future__ import annotations

import abc

import numpy


class Model(abc.ABC):
    def __init__(self, model_path: str):
        import onnxruntime

        self._session = onnxruntime.InferenceSession(model_path)

    @property
    def session(self):
        return self._session

    def _infer(self, batch: numpy.ndarray) -> numpy.ndarray:
        name = self._session.get_inputs()[0].name
        return self._session.run(None, {name: batch})[0]

    @abc.abstractmethod
    def predict(self, tile: numpy.ndarray, **kwargs) -> numpy.ndarray:
        ...


class Yolo26Model(Model):
    def predict(self, tile: numpy.ndarray, *, slice_size: int | None = None, **kwargs) -> numpy.ndarray:
        arr = tile.astype(numpy.float32) / 255.0
        if slice_size:
            arr = _pad(arr, 1, slice_size)
            arr = _pad(arr, 2, slice_size)
        detections = self._infer(arr[numpy.newaxis, ...])
        return numpy.squeeze(detections[..., 0:5], axis=0)


def _pad(array: numpy.ndarray, axis: int, target: int) -> numpy.ndarray:
    short = target - array.shape[axis]
    if short <= 0:
        return array
    width = [(0, 0)] * array.ndim
    width[axis] = (0, short)
    return numpy.pad(array, width, mode="constant", constant_values=0)
