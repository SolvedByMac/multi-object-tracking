import numpy as np

from src.tracking.association import (
    associate_iou,
    iou,
    iou_cost_matrix,
)


def test_iou_identical_boxes():
    box = np.array([10, 20, 50, 80], dtype=float)

    assert iou(box, box) == 1.0


def test_iou_non_overlapping_boxes():
    box_a = np.array([0, 0, 10, 10], dtype=float)
    box_b = np.array([20, 20, 30, 30], dtype=float)

    assert iou(box_a, box_b) == 0.0


def test_iou_cost_matrix_known_values():
    tracks = np.array(
        [
            [0, 0, 10, 10],
            [20, 20, 30, 30],
        ],
        dtype=float,
    )

    detections = np.array(
        [
            [0, 0, 10, 10],
            [20, 20, 30, 30],
        ],
        dtype=float,
    )

    costs = iou_cost_matrix(tracks, detections)

    expected = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=float,
    )

    np.testing.assert_allclose(costs, expected)


def test_hungarian_assignment_known_answer():
    tracks = np.array(
        [
            [0, 0, 10, 10],
            [20, 20, 30, 30],
            [40, 40, 50, 50],
        ],
        dtype=float,
    )

    detections = np.array(
        [
            [21, 21, 31, 31],
            [41, 41, 51, 51],
            [1, 1, 11, 11],
        ],
        dtype=float,
    )

    matches, unmatched_tracks, unmatched_detections = associate_iou(
        tracks,
        detections,
        min_iou=0.3,
    )

    assert sorted(matches) == [
        (0, 2),
        (1, 0),
        (2, 1),
    ]

    assert unmatched_tracks == []
    assert unmatched_detections == []


def test_iou_gate_rejects_bad_match():
    tracks = np.array(
        [[0, 0, 10, 10]],
        dtype=float,
    )

    detections = np.array(
        [[8, 8, 18, 18]],
        dtype=float,
    )

    matches, unmatched_tracks, unmatched_detections = associate_iou(
        tracks,
        detections,
        min_iou=0.3,
    )

    assert matches == []
    assert unmatched_tracks == [0]
    assert unmatched_detections == [0]
