"""Ship characterisation -- length, width, speed, course and type -- from a
chip centred on a detected ship.

The model (:class:`~satimg.prediction.models.CharacterisationModel`) returns
raw logits; this module decodes them. Each of length / width / sog / cog is a
softmax over fixed bins ("knots"), decoded as the expected knot value (a
circular mean for cog); the ship type is the argmax over :data:`SHIP_TYPES`.

The predicted cog is the course *as it points in the image*. That is the
compass course for a north-up optical product, but a Sentinel-1 GRD is a
mirror image of the map, so there it is converted back through the product's
``heading_in_image`` (its own inverse).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy

from satimg.patch import Patch
from satimg.prediction.models import CharacterisationModel
from satimg.prediction.types import Characterisation, Label
from satimg.product import Product

logger = logging.getLogger(__name__)

TASKS = ("length", "width", "sog", "cog")

#: the ship-type head's classes, in logit order.
SHIP_TYPES = ("cargo", "tanker", "fishing", "passenger", "leisure", "other")

#: one representative ITU-R M.1371 AIS ship type code per class.
SHIP_TYPE_CODES = {"cargo": 70, "tanker": 80, "fishing": 30, "passenger": 60, "leisure": 36, "other": 0}

#: (lo, hi) per task: metres, metres, m/s, degrees.
_RANGES = {
    "length": (0.0, 400.0),
    "width": (0.0, 60.0),
    "sog": (0.0, 15.43332),
    "cog": (0.0, 360.0),
}

_N_BINS = {
    "length": {"fine": 200, "medium": 40, "coarse": 20},
    "width": {"fine": 30, "medium": 6, "coarse": 3},
    "sog": {"fine": 30, "medium": 10, "coarse": 5},
    "cog": {"fine": 360, "medium": 24, "coarse": 12},
}


@dataclass
class CharacteriserConfig:
    chip_size: int = 256
    bin_size: str = "medium"


def knots(task: str, bin_size: str) -> numpy.ndarray:
    """The values ``task``'s bins stand for: ``n + 1`` evenly spaced over the
    range, or ``n`` around the circle (no endpoint) for cog."""
    lo, hi = _RANGES[task]
    n = _N_BINS[task][bin_size]
    if task == "cog":
        return numpy.linspace(lo, hi, n, endpoint=False)
    return numpy.linspace(lo, hi, n + 1)


def decode(logits: numpy.ndarray, bin_size: str = "medium") -> Characterisation:
    """One chip's ``(num_outputs,)`` logits -> :class:`Characterisation`, cog
    still in the image's own frame."""
    widths = [len(knots(task, bin_size)) for task in TASKS]
    expected = sum(widths) + len(SHIP_TYPES)
    if logits.shape != (expected,):
        raise ValueError(f"expected {expected} logits for bin_size={bin_size!r}, got shape {logits.shape}")

    values = {}
    start = 0
    for task, width in zip(TASKS, widths):
        probs = _softmax(logits[start:start + width])
        start += width
        k = knots(task, bin_size)
        if task == "cog":
            theta = numpy.deg2rad(k)
            values[task] = float(numpy.rad2deg(numpy.arctan2(probs @ numpy.sin(theta), probs @ numpy.cos(theta))) % 360.0)
        else:
            values[task] = float(probs @ k)

    type_probs = _softmax(logits[start:])
    name = SHIP_TYPES[int(type_probs.argmax())]
    return Characterisation(
        **values,
        ship_type=Label(name, float(type_probs.max())),
        ship_type_code=SHIP_TYPE_CODES[name],
    )


class Characteriser:
    def __init__(self, product: Product, model: CharacterisationModel, **overrides):
        self._product = product
        self._model = model
        self._config = CharacteriserConfig(**overrides)

    @property
    def chip_size(self) -> int:
        """The patch size :meth:`characterise` expects -- pass it to
        :meth:`~satimg.product.Product.patches_at`."""
        return self._config.chip_size

    def characterise(self, patch: Patch) -> Characterisation:
        """Characterise the ship at the centre of ``patch``, a
        :attr:`chip_size` patch from ``patches_at``. cog is returned in
        degrees clockwise from true north."""
        result = decode(self._model.predict(patch.visual.values), self._config.bin_size)
        # only SAR products define heading_in_image; optical ones are north-up
        if hasattr(self._product, "heading_in_image"):
            col, row = patch.center
            result.cog = float(self._product.heading_in_image((row, col), result.cog))
        return result


def _softmax(x: numpy.ndarray) -> numpy.ndarray:
    e = numpy.exp(x - x.max())
    return e / e.sum()
