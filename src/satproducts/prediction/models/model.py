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

    def _infer(self, subset, *args, **kwargs):
        input_name = self.session.get_inputs()[0].name
        detections = self.session.run(None, {input_name: subset})[0]
        return detections
