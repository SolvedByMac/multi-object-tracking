from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

CHI2_95_4DOF = 9.4877


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)

    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0.0:
        return 0.0

    return float(intersection / union)


def iou_cost_matrix(
    track_boxes: np.ndarray,
    detection_boxes: np.ndarray,
) -> np.ndarray:

    num_tracks = len(track_boxes)
    num_detections = len(detection_boxes)

    cost_matrix = np.empty(
        (num_tracks, num_detections),
        dtype=float,
    )

    for track_idx, track_box in enumerate(track_boxes):
        for detection_idx, detection_box in enumerate(detection_boxes):
            cost_matrix[track_idx, detection_idx] = 1.0 - iou(track_box, detection_box)

    return cost_matrix


def associate_iou(
    track_boxes: np.ndarray,
    detection_boxes: np.ndarray,
    min_iou: float = 0.3,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:

    num_tracks = len(track_boxes)
    num_detections = len(detection_boxes)

    if num_tracks == 0:
        return [], [], list(range(num_detections))

    if num_detections == 0:
        return [], list(range(num_tracks)), []

    cost_matrix = iou_cost_matrix(
        track_boxes,
        detection_boxes,
    )

    row_indices, column_indices = linear_sum_assignment(cost_matrix)

    matches: list[tuple[int, int]] = []
    matched_tracks: set[int] = set()
    matched_detections: set[int] = set()

    for track_idx, detection_idx in zip(
        row_indices,
        column_indices,
        strict=True,
    ):
        overlap = 1.0 - cost_matrix[track_idx, detection_idx]

        if overlap < min_iou:
            continue

        matches.append((track_idx, detection_idx))
        matched_tracks.add(track_idx)
        matched_detections.add(detection_idx)

    unmatched_tracks = [idx for idx in range(num_tracks) if idx not in matched_tracks]

    unmatched_detections = [
        idx for idx in range(num_detections) if idx not in matched_detections
    ]

    return matches, unmatched_tracks, unmatched_detections
