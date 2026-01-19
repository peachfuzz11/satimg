import os
from enum import Enum
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class ModelCatalog(Enum):
    S1 = "s1.onnx"
    S2 = "s2.onnx"

    def get_path(self):
        return os.path.join(BASE_DIR, "resources", self.value)
