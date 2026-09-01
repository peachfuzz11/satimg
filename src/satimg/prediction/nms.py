import numpy


def iou(boxes1, boxes2):
    """
    Calculate IoU matrix between two sets of bounding boxes.

    Args:
        boxes1: Bounding boxes in the format nx4 (x_center, y_center, w, h).
        boxes2: Bounding boxes in the format mx4 (x_center, y_center, w, h).

    Returns:
        IoU matrix with shape (n, m) containing IoU values for each pair of bounding boxes.
    """
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return numpy.empty((boxes1.shape[0], boxes2.shape[0]))
    # Convert from (x_center, y_center, w, h) to (x1, y1, x2, y2)
    x1_1 = boxes1[:, 0] - boxes1[:, 2] / 2
    y1_1 = boxes1[:, 1] - boxes1[:, 3] / 2
    x2_1 = boxes1[:, 0] + boxes1[:, 2] / 2
    y2_1 = boxes1[:, 1] + boxes1[:, 3] / 2

    x1_2 = boxes2[:, 0] - boxes2[:, 2] / 2
    y1_2 = boxes2[:, 1] - boxes2[:, 3] / 2
    x2_2 = boxes2[:, 0] + boxes2[:, 2] / 2
    y2_2 = boxes2[:, 1] + boxes2[:, 3] / 2

    # Calculate the intersection coordinates
    intersection_x1 = numpy.maximum(x1_1[:, None], x1_2)
    intersection_y1 = numpy.maximum(y1_1[:, None], y1_2)
    intersection_x2 = numpy.minimum(x2_1[:, None], x2_2)
    intersection_y2 = numpy.minimum(y2_1[:, None], y2_2)

    # Compute the intersection area
    intersection_area = numpy.maximum(intersection_x2 - intersection_x1, 0) * numpy.maximum(
        intersection_y2 - intersection_y1, 0)

    # Compute the area of each box
    area_boxes1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area_boxes2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Compute IoU: intersection_area / (area1 + area2 - intersection_area)
    return intersection_area / (area_boxes1[:, None] + area_boxes2 - intersection_area)


def non_max_suppression(confidences, bboxes, overlap_threshold=.3):
    """
    Perform non-maximum suppression on the input boxes using the given threshold.
    :param confidences: a numpy array of shape (N, K) containing confidences for K classes.
    :param bboxes: a numpy array of shape (N, 4) containing box coordinates in the format (x, y, w, h)
    :param overlap_threshold: the threshold value for suppression

    :return: an array of shape N containing the indices of the boxes to keep
    """
    keep = numpy.ones(bboxes.shape[0], dtype=bool)
    if bboxes.shape[0] < 2:
        return keep
    # Each box is assigned its maximum confidence class
    bbox_cls = numpy.argmax(confidences, axis=-1)

    # Loop through the classes
    for cls in range(confidences.shape[-1]):
        # Create a mask for the bboxes with specific class
        cls_mask = bbox_cls == cls
        cls_indices = numpy.nonzero(cls_mask)[0]  # Indices of bboxes with class cls

        masked_bboxes = bboxes[cls_mask]  # Now size (C,4)
        masked_confidences = confidences[cls_mask, cls]  # Now size (C,1)

        sort = numpy.argsort(masked_confidences, axis=0)[::-1]  # Sort based on confidence
        sorted_bboxes = masked_bboxes[sort]
        masked_indices = cls_indices[sort]

        iou_matrix = iou(sorted_bboxes, sorted_bboxes)
        # Apply threshold
        iou_matrix = iou_matrix >= overlap_threshold
        # The iou matrix now contains True where two bboxes overlap more than allowed.
        # Since the matrix is sorted by confidence, the first bbox always suppresses the others

        # Zero out diagonal and upper triangular part
        iou_matrix = iou_matrix * numpy.tril(iou_matrix, -1)
        # The matrix is now such that if theres a true on any row the corresponding box must be suppressed
        suppressed = numpy.any(iou_matrix, axis=1)
        suppressed_indexes = masked_indices[suppressed]
        keep[suppressed_indexes] = False
    return keep


if __name__ == '__main__':
    import numpy as np

    # Case 1: Non-overlapping boxes (basic case)
    confidences = np.array([[0.9], [0.8], [0.75]])
    bboxes = np.array([[10, 10, 20, 20], [100, 100, 20, 20], [200, 200, 20, 20]])  # No overlap
    overlap_threshold = 0.3
    keep_indices = non_max_suppression(confidences, bboxes, overlap_threshold)
    assert numpy.all(keep_indices), f"Test failed for non-overlapping boxes: {keep_indices}"

    # Case 2: Fully overlapping boxes, keep highest confidence
    confidences = np.array([[0.9], [0.8]])
    bboxes = np.array([[10, 10, 20, 20], [10, 10, 20, 20]])  # Full overlap
    overlap_threshold = 0.3
    keep_indices = non_max_suppression(confidences, bboxes, overlap_threshold)
    expected_indices = np.array([True, False])  # Keep the box with the highest confidence
    assert np.array_equal(keep_indices,
                          expected_indices), f"Test failed for fully overlapping boxes: {keep_indices}"

    # Case 3: Partially overlapping boxes
    confidences = np.array([[0.9], [0.95]])
    bboxes = np.array([[10, 10, 20, 20], [10, 10, 20, 20]])  # Partial overlap
    overlap_threshold = 0.5  # Lower threshold for partial overlap
    keep_indices = non_max_suppression(confidences, bboxes, overlap_threshold)
    expected_indices = np.array([False, True])  # Keep box 0, suppress box 1
    assert np.array_equal(keep_indices,
                          expected_indices), f"Test failed for partially overlapping boxes: {keep_indices}"

    # Case 4: All boxes suppressed
    confidences = np.array([[0.6], [0.7], [0.5]])
    bboxes = np.array([[10, 10, 20, 20], [10, 10, 20, 20], [10, 10, 20, 20]])  # Fully overlapping boxes
    overlap_threshold = 0.3
    keep_indices = non_max_suppression(confidences, bboxes, overlap_threshold)
    expected_indices = np.array([False, True, False])  # Only keep the box with the highest confidence (index 1)
    assert np.array_equal(keep_indices, expected_indices), f"Test failed for all suppressed case: {keep_indices}"

    # Case 5: Mixed overlap, some boxes suppressed, some kept
    confidences = np.array([[0.9], [0.7], [0.8]])
    bboxes = np.array([[10, 10, 20, 20], [12, 12, 20, 20], [30, 30, 20, 20]])  # Mixed overlap and no overlap
    overlap_threshold = 0.3
    keep_indices = non_max_suppression(confidences, bboxes, overlap_threshold)
    expected_indices = np.array([True, False, True])  # Box 1 should be suppressed, 0 and 2 kept
    assert np.array_equal(keep_indices, expected_indices), f"Test failed for mixed overlap: {keep_indices}"

    # Case 6: Single box (trivial case)
    confidences = np.array([[0.5]])
    bboxes = np.array([[10, 10, 20, 20]])
    overlap_threshold = 0.3
    keep_indices = non_max_suppression(confidences, bboxes, overlap_threshold)
    expected_indices = np.array([True])  # Only one box, should be kept
    assert np.array_equal(keep_indices, expected_indices), f"Test failed for single box case: {keep_indices}"

    # Case 7: High overlap threshold, allowing boxes with high overlap to stay
    confidences = np.array([[0.9], [0.85]])
    bboxes = np.array([[10, 10, 20, 20], [15, 15, 20, 20]])  # Partial overlap
    overlap_threshold = 0.3
    keep_indices = non_max_suppression(confidences, bboxes, overlap_threshold)
    expected_indices = np.array([True, False])  # Both boxes should be kept
    assert np.array_equal(keep_indices, expected_indices), f"Test failed for high overlap threshold: {keep_indices}"
    print("All tests passed!")
