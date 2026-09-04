from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.tracking.kalman import KalmanFilter

CHI2_95_4DOF = 9.4877
GATED_COST = 1e6


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
            cost_matrix[track_idx, detection_idx] = 1.0 - iou(
                track_box,
                detection_box,
            )

    return cost_matrix


def cosine_cost_matrix(
    track_embeddings: np.ndarray,
    detection_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Compute pairwise cosine distances.

    Inputs are expected to contain L2-normalized embeddings.
    """
    track_embeddings = np.asarray(
        track_embeddings,
        dtype=float,
    )

    detection_embeddings = np.asarray(
        detection_embeddings,
        dtype=float,
    )

    if track_embeddings.ndim != 2:
        raise ValueError("track_embeddings must be a 2D array")

    if detection_embeddings.ndim != 2:
        raise ValueError("detection_embeddings must be a 2D array")

    if track_embeddings.shape[1] != detection_embeddings.shape[1]:
        raise ValueError("embedding dimensions must match")

    similarities = track_embeddings @ detection_embeddings.T

    similarities = np.clip(
        similarities,
        -1.0,
        1.0,
    )

    return 1.0 - similarities


def xyxy_to_cxcywh_array(
    boxes: np.ndarray,
) -> np.ndarray:
    boxes = np.asarray(
        boxes,
        dtype=float,
    )

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4)")

    if len(boxes) == 0:
        return np.empty((0, 4), dtype=float)

    measurements = np.empty_like(boxes)

    measurements[:, 2] = boxes[:, 2] - boxes[:, 0]

    measurements[:, 3] = boxes[:, 3] - boxes[:, 1]

    measurements[:, 0] = boxes[:, 0] + measurements[:, 2] / 2.0

    measurements[:, 1] = boxes[:, 1] + measurements[:, 3] / 2.0

    return measurements


def fused_cost_matrix(
    track_boxes: np.ndarray,
    detection_boxes: np.ndarray,
    track_embeddings: np.ndarray,
    detection_embeddings: np.ndarray,
    lambda_motion: float = 0.98,
) -> np.ndarray:

    if not 0.0 <= lambda_motion <= 1.0:
        raise ValueError("lambda_motion must be between 0 and 1")

    motion_cost = iou_cost_matrix(
        track_boxes,
        detection_boxes,
    )

    appearance_cost = cosine_cost_matrix(
        track_embeddings,
        detection_embeddings,
    )

    return lambda_motion * motion_cost + (1.0 - lambda_motion) * appearance_cost


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
        overlap = (
            1.0
            - cost_matrix[
                track_idx,
                detection_idx,
            ]
        )

        if overlap < min_iou:
            continue

        matches.append((track_idx, detection_idx))

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


def associate_fused(
    track_boxes: np.ndarray,
    track_means: np.ndarray,
    track_covariances: np.ndarray,
    track_embeddings: np.ndarray,
    detection_boxes: np.ndarray,
    detection_embeddings: np.ndarray,
    kf: KalmanFilter,
    lambda_motion: float = 0.98,
    gating_threshold: float = CHI2_95_4DOF,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:

    num_tracks = len(track_boxes)
    num_detections = len(detection_boxes)

    if num_tracks == 0:
        return [], [], list(range(num_detections))

    if num_detections == 0:
        return [], list(range(num_tracks)), []

    if len(track_means) != num_tracks:
        raise ValueError("track_means must align with track_boxes")

    if len(track_covariances) != num_tracks:
        raise ValueError("track_covariances must align with track_boxes")

    if len(track_embeddings) != num_tracks:
        raise ValueError("track_embeddings must align with track_boxes")

    if len(detection_embeddings) != num_detections:
        raise ValueError("detection_embeddings must align with detection_boxes")

    cost_matrix = fused_cost_matrix(
        track_boxes=track_boxes,
        detection_boxes=detection_boxes,
        track_embeddings=track_embeddings,
        detection_embeddings=detection_embeddings,
        lambda_motion=lambda_motion,
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
        if (
            cost_matrix[
                track_idx,
                detection_idx,
            ]
            >= GATED_COST
        ):
            continue

        matches.append((track_idx, detection_idx))

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
