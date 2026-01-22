from dataclasses import dataclass


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class Label:
    label: str
    conf: float


@dataclass
class Coordinate:
    lat: float
    lon: float


@dataclass
class Detection:
    bbox: BBox
    label: Label
    coordinate: Coordinate
