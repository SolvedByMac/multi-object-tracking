import numpy as np

from src.tracking.association import (
    associate_fused,
    associate_iou,
    cosine_cost_matrix,
    fused_cost_matrix,
    iou,
    iou_cost_matrix,
    xyxy_to_cxcywh_array,
)
from src.tracking.kalman import KalmanFilter


def test_iou_identical_boxes():
    box = np.array(
        [10, 20, 50, 80],
        dtype=float,
    )

    assert iou(box, box) == 1.0


def test_iou_non_overlapping_boxes():
    box_a = np.array(
        [0, 0, 10, 10],
        dtype=float,
    )

    box_b = np.array(
        [20, 20, 30, 30],
        dtype=float,
    )

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

    costs = iou_cost_matrix(
        tracks,
        detections,
    )

    expected = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=float,
    )

    np.testing.assert_allclose(
        costs,
        expected,
    )


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

    (
        matches,
        unmatched_tracks,
        unmatched_detections,
    ) = associate_iou(
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

    (
        matches,
        unmatched_tracks,
        unmatched_detections,
    ) = associate_iou(
        tracks,
        detections,
        min_iou=0.3,
    )

    assert matches == []
    assert unmatched_tracks == [0]
    assert unmatched_detections == [0]


def test_cosine_cost_matrix_identical_embeddings():
    tracks = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )

    detections = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )

    costs = cosine_cost_matrix(
        tracks,
        detections,
    )

    expected = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=float,
    )

    np.testing.assert_allclose(
        costs,
        expected,
    )


def test_fused_cost_known_values():
    track_boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
        ]
    )

    detection_boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
        ]
    )

    track_embeddings = np.array(
        [
            [1.0, 0.0],
        ]
    )

    detection_embeddings = np.array(
        [
            [0.0, 1.0],
        ]
    )

    costs = fused_cost_matrix(
        track_boxes,
        detection_boxes,
        track_embeddings,
        detection_embeddings,
        lambda_motion=0.98,
    )

    np.testing.assert_allclose(
        costs,
        [[0.02]],
        atol=1e-8,
    )


def test_xyxy_to_cxcywh_array():
    boxes = np.array(
        [
            [
                10.0,
                20.0,
                30.0,
                60.0,
            ],
        ]
    )

    measurements = xyxy_to_cxcywh_array(boxes)

    np.testing.assert_allclose(
        measurements,
        [
            [
                20.0,
                40.0,
                20.0,
                40.0,
            ]
        ],
    )


def test_fused_association_accepts_plausible_match():
    kf = KalmanFilter()

    measurement = np.array(
        [
            50.0,
            100.0,
            20.0,
            40.0,
        ],
        dtype=float,
    )

    mean, covariance = kf.initiate(measurement)

    track_boxes = np.array(
        [
            [
                40.0,
                80.0,
                60.0,
                120.0,
            ]
        ]
    )

    detection_boxes = np.array(
        [
            [
                41.0,
                81.0,
                61.0,
                121.0,
            ]
        ]
    )

    track_embeddings = np.array(
        [
            [1.0, 0.0],
        ]
    )

    detection_embeddings = np.array(
        [
            [1.0, 0.0],
        ]
    )

    (
        matches,
        unmatched_tracks,
        unmatched_detections,
    ) = associate_fused(
        track_boxes=track_boxes,
        track_means=np.array([mean]),
        track_covariances=np.array([covariance]),
        track_embeddings=track_embeddings,
        detection_boxes=detection_boxes,
        detection_embeddings=detection_embeddings,
        kf=kf,
    )

    assert matches == [(0, 0)]
    assert unmatched_tracks == []
    assert unmatched_detections == []


def test_mahalanobis_gate_rejects_far_detection():
    kf = KalmanFilter()

    measurement = np.array(
        [
            50.0,
            100.0,
            20.0,
            40.0,
        ],
        dtype=float,
    )

    mean, covariance = kf.initiate(measurement)

    track_boxes = np.array(
        [
            [
                40.0,
                80.0,
                60.0,
                120.0,
            ]
        ]
    )

    detection_boxes = np.array(
        [
            [
                400.0,
                400.0,
                420.0,
                440.0,
            ]
        ]
    )

    track_embeddings = np.array(
        [
            [1.0, 0.0],
        ]
    )

    detection_embeddings = np.array(
        [
            [1.0, 0.0],
        ]
    )

    (
        matches,
        unmatched_tracks,
        unmatched_detections,
    ) = associate_fused(
        track_boxes=track_boxes,
        track_means=np.array([mean]),
        track_covariances=np.array([covariance]),
        track_embeddings=track_embeddings,
        detection_boxes=detection_boxes,
        detection_embeddings=detection_embeddings,
        kf=kf,
    )

    assert matches == []
    assert unmatched_tracks == [0]
    assert unmatched_detections == [0]
