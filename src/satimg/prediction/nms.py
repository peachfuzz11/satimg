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
