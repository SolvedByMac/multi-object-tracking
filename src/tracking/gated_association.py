from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.tracking.association import (
    CHI2_95_4DOF,
    GATED_COST,
    iou,
    iou_cost_matrix,
    xyxy_to_cxcywh_array,
)
from src.tracking.kalman import KalmanFilter


def associate_iou_gated(
    track_boxes: np.ndarray,
    track_means: np.ndarray,
    track_covariances: np.ndarray,
    detection_boxes: np.ndarray,
    kf: KalmanFilter,
    min_iou: float = 0.3,
    gating_threshold: float = CHI2_95_4DOF,
) -> tuple[
    list[tuple[int, int]],
    list[int],
    list[int],
]:
    num_tracks = len(track_boxes)
    num_detections = len(detection_boxes)

    if num_tracks == 0:
        return (
            [],
            [],
            list(range(num_detections)),
        )

    if num_detections == 0:
        return (
            [],
            list(range(num_tracks)),
            [],
        )

    if len(track_means) != num_tracks:
        raise ValueError("track_means must align with track_boxes")

    if len(track_covariances) != num_tracks:
        raise ValueError("track_covariances must align with track_boxes")

    cost_matrix = iou_cost_matrix(
        track_boxes,
        detection_boxes,
    )

    measurements = xyxy_to_cxcywh_array(detection_boxes)

    for track_idx in range(num_tracks):
        distances = kf.gating_distance(
            track_means[track_idx],
            track_covariances[track_idx],
            measurements,
        )

        gated = distances > gating_threshold

        cost_matrix[
            track_idx,
            gated,
        ] = GATED_COST

    row_indices, column_indices = linear_sum_assignment(cost_matrix)

    matches: list[tuple[int, int]] = []
    matched_tracks: set[int] = set()
    matched_detections: set[int] = set()

    for track_idx, detection_idx in zip(
        row_indices,
        column_indices,
        strict=True,
    ):
        if cost_matrix[track_idx, detection_idx] >= GATED_COST:
            continue

        overlap = iou(
            track_boxes[track_idx],
            detection_boxes[detection_idx],
        )

        if overlap < min_iou:
            continue

        matches.append(
            (
                track_idx,
                detection_idx,
            )
        )

        matched_tracks.add(track_idx)
        matched_detections.add(detection_idx)

    unmatched_tracks = [idx for idx in range(num_tracks) if idx not in matched_tracks]

    unmatched_detections = [
        idx for idx in range(num_detections) if idx not in matched_detections
    ]

    return (
        matches,
        unmatched_tracks,
        unmatched_detections,
    )
