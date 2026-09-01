import numpy
import pytest

from satimg.prediction import nms


@pytest.mark.parametrize(
    "confidences, bboxes, threshold, expected",
    [
        ([[0.9], [0.8], [0.75]],
         [[10, 10, 20, 20], [100, 100, 20, 20], [200, 200, 20, 20]], 0.3,
         [True, True, True]),
        ([[0.9], [0.8]], [[10, 10, 20, 20], [10, 10, 20, 20]], 0.3, [True, False]),
        ([[0.9], [0.95]], [[10, 10, 20, 20], [10, 10, 20, 20]], 0.5, [False, True]),
        ([[0.6], [0.7], [0.5]],
         [[10, 10, 20, 20], [10, 10, 20, 20], [10, 10, 20, 20]], 0.3,
         [False, True, False]),
        ([[0.9], [0.7], [0.8]],
         [[10, 10, 20, 20], [12, 12, 20, 20], [30, 30, 20, 20]], 0.3,
         [True, False, True]),
        ([[0.5]], [[10, 10, 20, 20]], 0.3, [True]),
    ],
)
def test_non_max_suppression(confidences, bboxes, threshold, expected):
    keep = nms.non_max_suppression(
        numpy.array(confidences, dtype=float),
        numpy.array(bboxes, dtype=float),
        overlap_threshold=threshold,
    )
    assert keep.tolist() == expected


def test_iou_empty():
    out = nms.iou(numpy.empty((0, 4)), numpy.ones((3, 4)))
    assert out.shape == (0, 3)


def test_iou_identical_boxes_is_one():
    box = numpy.array([[0.0, 0.0, 10.0, 10.0]])
    assert nms.iou(box, box)[0, 0] == pytest.approx(1.0)
