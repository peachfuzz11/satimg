from types import SimpleNamespace

import numpy
import pytest

from satimg.prediction import CharacterisationModel, Characteriser
from satimg.prediction.characteriser import SHIP_TYPES, TASKS, bin_values, decode

#: logits wide enough to put ~all softmax mass on one bin
_PEAK = 50.0


def _logits(peaks: dict[str, list[int]], ship_type: int = 0) -> numpy.ndarray:
    """Flat logits with a peak at each listed bin index per task."""
    parts = []
    for task in TASKS:
        part = numpy.zeros(len(bin_values(task)))
        part[peaks.get(task, [0])] = _PEAK
        parts.append(part)
    types = numpy.zeros(len(SHIP_TYPES))
    types[ship_type] = _PEAK
    return numpy.concatenate([*parts, types])


def test_output_width():
    assert sum(len(bin_values(t)) for t in TASKS) + len(SHIP_TYPES) == 89


def test_decodes_peaked_bins_to_their_bin_values():
    result = decode(_logits({"length": [10], "width": [2], "sog": [4], "cog": [6]}, ship_type=2))

    assert result.length == pytest.approx(100.0)
    assert result.width == pytest.approx(20.0)
    assert result.sog == pytest.approx(12.0)  # knots
    assert result.cog == pytest.approx(90.0)
    assert result.ship_type.name == "fishing"
    assert result.ship_type.confidence == pytest.approx(1.0)
    assert result.ship_type_code == 30


def test_splits_mass_between_neighbouring_bins_linearly():
    result = decode(_logits({"length": [10, 11]}))
    assert result.length == pytest.approx(105.0)


def test_cog_is_a_circular_mean_across_north():
    result = decode(_logits({"cog": [23, 1]}))  # 345 deg and 15 deg
    assert min(result.cog, 360.0 - result.cog) == pytest.approx(0.0, abs=1e-6)


def test_rejects_logits_of_the_wrong_width():
    with pytest.raises(ValueError):
        decode(numpy.zeros(10))


class _FakeModel:
    def __init__(self, logits):
        self.logits = logits
        self.tiles = []

    def predict(self, tile):
        self.tiles.append(tile)
        return self.logits


def _patch():
    visual = SimpleNamespace(values=numpy.zeros((3, 64, 64), dtype=numpy.uint8))
    return SimpleNamespace(visual=visual, center=(30.0, 40.0))


def test_optical_product_keeps_cog_as_predicted():
    model = _FakeModel(_logits({"cog": [6]}))
    result = Characteriser(SimpleNamespace(), model).characterise(_patch())
    assert result.cog == pytest.approx(90.0)
    assert model.tiles[0].shape == (3, 64, 64)


def test_sar_product_converts_cog_out_of_the_image_frame():
    calls = []

    def heading_from_image(rowcol, heading):
        calls.append((rowcol, heading))
        return 123.0

    product = SimpleNamespace(heading_from_image=heading_from_image)
    result = Characteriser(product, _FakeModel(_logits({"cog": [6]}))).characterise(_patch())

    assert result.cog == 123.0
    (rowcol, heading), = calls
    assert rowcol == (40.0, 30.0)
    assert heading == pytest.approx(90.0)


def test_model_repeats_a_single_band_chip_to_rgb():
    model = CharacterisationModel.__new__(CharacterisationModel)
    seen = []
    model._infer = lambda batch: seen.append(batch) or numpy.zeros((1, 89))

    model.predict(numpy.ones((1, 64, 64), dtype=numpy.uint8))

    assert seen[0].shape == (1, 3, 64, 64)
    assert seen[0].dtype == numpy.uint8
