import numpy as np

from src.tracking.gated_association import associate_iou_gated
from src.tracking.kalman import KalmanFilter


def test_iou_gated_accepts_valid_match():
    kf = KalmanFilter()

    measurement = np.array(
        [
            50.0,
            100.0,
            40.0,
            100.0,
        ]
    )

    mean, covariance = kf.initiate(measurement)

    track_boxes = np.array(
        [
            [
                30.0,
                50.0,
                70.0,
                150.0,
            ]
        ]
    )

    detection_boxes = np.array(
        [
            [
                32.0,
                51.0,
                72.0,
                151.0,
            ]
        ]
    )

    (
        matches,
        unmatched_tracks,
        unmatched_detections,
    ) = associate_iou_gated(
        track_boxes=track_boxes,
        track_means=np.array([mean]),
        track_covariances=np.array([covariance]),
        detection_boxes=detection_boxes,
        kf=kf,
        min_iou=0.3,
    )

    assert matches == [(0, 0)]
    assert unmatched_tracks == []
    assert unmatched_detections == []


def test_iou_gated_rejects_mahalanobis_outlier():
    kf = KalmanFilter()

    measurement = np.array(
        [
            50.0,
            100.0,
            40.0,
            100.0,
        ]
    )

    mean, covariance = kf.initiate(measurement)

    track_boxes = np.array(
        [
            [
                30.0,
                50.0,
                70.0,
                150.0,
            ]
        ]
    )

    detection_boxes = np.array(
        [
            [
                32.0,
                51.0,
                72.0,
                151.0,
            ]
        ]
    )

    (
        matches,
        unmatched_tracks,
        unmatched_detections,
    ) = associate_iou_gated(
        track_boxes=track_boxes,
        track_means=np.array([mean]),
        track_covariances=np.array([covariance]),
        detection_boxes=detection_boxes,
        kf=kf,
        min_iou=0.3,
        gating_threshold=-1.0,
    )

    assert matches == []
    assert unmatched_tracks == [0]
    assert unmatched_detections == [0]


def test_iou_gated_handles_no_tracks():
    kf = KalmanFilter()

    matches, unmatched_tracks, unmatched_detections = associate_iou_gated(
        track_boxes=np.empty((0, 4)),
        track_means=np.empty((0, 8)),
        track_covariances=np.empty((0, 8, 8)),
        detection_boxes=np.array(
            [
                [
                    30.0,
                    50.0,
                    70.0,
                    150.0,
                ]
            ]
        ),
        kf=kf,
    )

    assert matches == []
    assert unmatched_tracks == []
    assert unmatched_detections == [0]


def test_iou_gated_handles_no_detections():
    kf = KalmanFilter()

    measurement = np.array(
        [
            50.0,
            100.0,
            40.0,
            100.0,
        ]
    )

    mean, covariance = kf.initiate(measurement)

    matches, unmatched_tracks, unmatched_detections = associate_iou_gated(
        track_boxes=np.array(
            [
                [
                    30.0,
                    50.0,
                    70.0,
                    150.0,
                ]
            ]
        ),
        track_means=np.array([mean]),
        track_covariances=np.array([covariance]),
        detection_boxes=np.empty((0, 4)),
        kf=kf,
    )

    assert matches == []
    assert unmatched_tracks == [0]
    assert unmatched_detections == []
