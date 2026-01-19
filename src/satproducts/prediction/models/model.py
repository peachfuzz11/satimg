import abc

import onnxruntime


class Model(abc.ABC):

    def __init__(self, model_path):
        session = onnxruntime.InferenceSession(model_path)
        self._session = session

    @property
    def session(self):
        return self._session

    def predict(self, *args, **kwargs):
        return self._infer(*args, **kwargs)

    @abc.abstractmethod
    def _infer(self, *args, **kwargs):
        pass
