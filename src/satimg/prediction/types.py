from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BBox:
    """Axis-aligned box in full-image pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2


@dataclass
class Label:
    name: str
    confidence: float


@dataclass
class Coordinate:
    lat: float
    lon: float


@dataclass
class Detection:
    bbox: BBox
    label: Label
    coordinate: Coordinate


@dataclass
class Characterisation:
    length: float  # metres
    width: float  # metres
    sog: float  # knots
    cog: float  # degrees clockwise from true north
    ship_type: Label
    ship_type_code: int  # a representative AIS ship type code for ship_type
