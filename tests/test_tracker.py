import numpy as np

from src.detection.detector import Detection
from src.tracking.tracker import Tracker, TrackerConfig


def make_detection(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    confidence: float = 0.9,
) -> Detection:
    return Detection(
        xyxy=np.array([x1, y1, x2, y2], dtype=float),
        confidence=confidence,
    )


def test_tracker_creates_and_confirms_track():
    tracker = Tracker(
        TrackerConfig(
            min_iou=0.3,
            n_init=2,
            max_age=30,
        )
    )

    detection_1 = make_detection(
        100,
        50,
        140,
        150,
    )

    confirmed = tracker.update([detection_1])

    assert confirmed == []

    detection_2 = make_detection(
        102,
        51,
        142,
        151,
    )

    confirmed = tracker.update([detection_2])

    assert len(confirmed) == 1
    assert confirmed[0].track_id == 1


def test_tracker_keeps_same_id_across_frames():
    tracker = Tracker(
        TrackerConfig(
            min_iou=0.3,
            n_init=1,
            max_age=30,
        )
    )

    detections = [
        make_detection(100, 50, 140, 150),
        make_detection(102, 51, 142, 151),
        make_detection(104, 52, 144, 152),
    ]

    ids = []

    for detection in detections:
        confirmed = tracker.update([detection])

        assert len(confirmed) == 1
        ids.append(confirmed[0].track_id)

    assert ids == [1, 1, 1]


def test_tracker_spawns_new_identity_for_distant_detection():
    tracker = Tracker(
        TrackerConfig(
            min_iou=0.3,
            n_init=1,
            max_age=30,
        )
    )

    first = tracker.update([make_detection(0, 0, 20, 40)])

    assert first[0].track_id == 1

    second = tracker.update([make_detection(200, 200, 220, 240)])

    assert len(second) == 1
    assert second[0].track_id == 2


def test_tracker_survives_short_missing_period():
    tracker = Tracker(
        TrackerConfig(
            min_iou=0.3,
            n_init=1,
            max_age=3,
        )
    )

    first = tracker.update([make_detection(100, 50, 140, 150)])

    assert first[0].track_id == 1

    tracker.update([])

    recovered = tracker.update([make_detection(102, 51, 142, 151)])

    assert len(recovered) == 1
    assert recovered[0].track_id == 1
