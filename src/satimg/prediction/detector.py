"""Sliced ship detection over a :class:`~satimg.product.Product`.

The product's ``visual`` view is walked in fixed-size, zero-padded tiles; each tile
is run through the model, boxes are filtered by size / aspect / confidence,
mapped back to full-image pixels via ``patch.window``, then to lat/lon, and
optionally dropped if they land on shore.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy

from satimg import tiling
from satimg.landmask import LandMask
from satimg.prediction import nms
from satimg.prediction.models import Model
from satimg.prediction.types import BBox, Coordinate, Detection, Label
from satimg.product import Product

logger = logging.getLogger(__name__)


@dataclass
class DetectorConfig:
    conf_threshold: float = 0.1
    nms_threshold: float | None = None
    slice_size: int = 256
    overlap: int = 0
    min_size: float = 1
    max_size: float = 300
    min_area: float = 4
    max_area: float = 300 ** 2
    max_aspect_ratio: float = 15
    apply_landmask: bool = True
    label: str = "ship"


class Detector:
    def __init__(self, product: Product, model: Model, **overrides):
        self._product = product
        self._model = model
        self._land = LandMask()
        self._config = DetectorConfig(**overrides)

    def __enter__(self) -> "Detector":
        return self

    def __exit__(self, *exc) -> None:
        self._product.__exit__(*exc)  # scope the product's rasters to the detector

    def detect(self, **overrides) -> list[Detection]:
        cfg = DetectorConfig(**{**self._config.__dict__, **overrides})
        logger.info(
            "detecting over %s: slice=%d overlap=%d",
            self._product,
            cfg.slice_size,
            cfg.overlap,
        )
        data = self._product.visual.chunk(
            {"x": cfg.slice_size, "y": cfg.slice_size, "band": -1}
        ).persist()
        results: list[Detection] = []

        for patch in tiling.patches(data, cfg.slice_size, overlap=cfg.overlap, edge="pad"):
            boxes = self._model.predict(patch.values, slice_size=cfg.slice_size)
            n_raw = len(boxes)
            boxes = boxes[boxes[:, 4] > cfg.conf_threshold]
            if cfg.nms_threshold:
                boxes = boxes[nms.non_max_suppression(
                    boxes[:, 4:], boxes[:, :4], overlap_threshold=cfg.nms_threshold
                )]
            boxes = boxes[_shape_mask(boxes, cfg)]
            logger.debug(
                "tile (%d,%d): %d boxes -> %d after filters",
                patch.window.col,
                patch.window.row,
                n_raw,
                len(boxes),
            )

            boxes[:, [0, 2]] += patch.window.col
            boxes[:, [1, 3]] += patch.window.row
            results.extend(self._to_detections(boxes, cfg))
        logger.info("detection complete: %d detection(s)", len(results))
        return results

    def _to_detections(self, boxes: numpy.ndarray, cfg: DetectorConfig) -> list[Detection]:
        out = []
        for row in boxes:
            x1, y1, x2, y2, conf = map(float, row)
            cx, cy = x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2
            lat, lon = self._product.transformer.rowcol_to_latlon((cy, cx))[0]
            if cfg.apply_landmask and self._land.contains(float(lon), float(lat)):
                continue
            out.append(
                Detection(
                    BBox(x1, y1, x2, y2),
                    Label(cfg.label, conf),
                    Coordinate(lat=float(lat), lon=float(lon)),
                )
            )
        return out


def _shape_mask(boxes: numpy.ndarray, cfg: DetectorConfig) -> numpy.ndarray:
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    area = w * h
    aspect = numpy.maximum(w / h, h / w)
    return (
        (cfg.min_size <= w) & (w <= cfg.max_size)
        & (cfg.min_size <= h) & (h <= cfg.max_size)
        & (cfg.min_area < area) & (area < cfg.max_area)
        & (aspect < cfg.max_aspect_ratio)
    )
