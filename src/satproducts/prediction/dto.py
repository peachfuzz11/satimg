from dataclasses import dataclass


@dataclass
class BBox:
    x: float
    y: float
    w: float
    h: float


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
