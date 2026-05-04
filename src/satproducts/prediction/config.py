from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_path: str


@dataclass
class DetectorConfig:
    label_mapping = {0: "ship", 1: "other", 2: "background"}
    conf_threshold: float = 0.25
    slice_size: int = 512
    min_size: float = 1
    max_size: float = 300
    min_area: float = 4
    max_area: float = 300 ** 2
    max_aspect_ratio: float = 15
    apply_landmask: bool = True
