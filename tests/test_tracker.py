import numpy as np

from src.detection.detector import Detection
from src.tracking.tracker import (
    Tracker,
    TrackerConfig,
)


def make_detection(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    confidence: float = 0.9,
) -> Detection:
    return Detection(
        xyxy=np.array(
            [x1, y1, x2, y2],
            dtype=float,
        ),
        confidence=confidence,
    )


def make_embedding(
    *values: float,
) -> np.ndarray:
    embedding = np.asarray(
        values,
        dtype=float,
    )

    norm = np.linalg.norm(embedding)

    if norm == 0.0:
        raise ValueError("embedding cannot be zero")

    return embedding / norm


def test_tracker_creates_and_confirms_track():
    tracker = Tracker(
        TrackerConfig(
            min_iou=0.3,
            n_init=2,
            max_age=30,
        )
    )

    first = tracker.update(
        [
            make_detection(
                100,
                50,
                140,
                150,
            )
        ]
    )

    assert first == []

    second = tracker.update(
        [
            make_detection(
                102,
                51,
                142,
                151,
            )
        ]
    )

    assert len(second) == 1
    assert second[0].track_id == 1


def test_tracker_keeps_same_id_across_frames():
    tracker = Tracker(
        TrackerConfig(
            min_iou=0.3,
            n_init=1,
            max_age=30,
        )
    )

    detections = [
        make_detection(
            100,
            50,
            140,
            150,
        ),
        make_detection(
            102,
            51,
            142,
            151,
        ),
        make_detection(
            104,
            52,
            144,
            152,
        ),
    ]

    ids = []

    for detection in detections:
        tracks = tracker.update([detection])

        assert len(tracks) == 1

        ids.append(tracks[0].track_id)

    assert ids == [1, 1, 1]


def test_tracker_spawns_new_identity_for_distant_detection():
    tracker = Tracker(
        TrackerConfig(
            min_iou=0.3,
            n_init=1,
            max_age=30,
        )
    )

    first = tracker.update(
        [
            make_detection(
                0,
                0,
                20,
                40,
            )
        ]
    )

    assert first[0].track_id == 1

    second = tracker.update(
        [
            make_detection(
                200,
                200,
                220,
                240,
            )
        ]
    )

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

    first = tracker.update(
        [
            make_detection(
                100,
                50,
                140,
                150,
            )
        ]
    )

    assert first[0].track_id == 1

    tracker.update([])

    recovered = tracker.update(
        [
            make_detection(
                102,
                51,
                142,
                151,
            )
        ]
    )

    assert len(recovered) == 1
    assert recovered[0].track_id == 1


def test_tracker_uses_appearance_embeddings():
    tracker = Tracker(
        TrackerConfig(
            min_iou=0.3,
            n_init=1,
            max_age=30,
        )
    )

    embedding = make_embedding(
        1.0,
        0.0,
    )

    first = tracker.update(
        [
            make_detection(
                100,
                50,
                140,
                150,
            )
        ],
        embeddings=np.array([embedding]),
    )

    assert len(first) == 1
    assert first[0].track_id == 1

    second = tracker.update(
        [
            make_detection(
                102,
                51,
                142,
                151,
            )
        ],
        embeddings=np.array([embedding]),
    )

    assert len(second) == 1
    assert second[0].track_id == 1


def test_tentative_track_uses_iou_fallback():
    tracker = Tracker(
        TrackerConfig(
            min_iou=0.3,
            n_init=2,
            max_age=30,
            max_cosine_distance=0.4415,
        )
    )

    first_embedding = make_embedding(
        1.0,
        0.0,
    )

    second_embedding = make_embedding(
        0.0,
        1.0,
    )

    first = tracker.update(
        [
            make_detection(
                100,
                50,
                140,
                150,
            )
        ],
        embeddings=np.array([first_embedding]),
    )

    assert first == []

    second = tracker.update(
        [
            make_detection(
                102,
                51,
                142,
                151,
            )
        ],
        embeddings=np.array([second_embedding]),
    )

    assert len(second) == 1
    assert second[0].track_id == 1


def test_recent_confirmed_track_uses_iou_fallback():
    tracker = Tracker(
        TrackerConfig(
            min_iou=0.3,
            n_init=1,
            max_age=30,
            max_cosine_distance=0.4415,
        )
    )

    first_embedding = make_embedding(
        1.0,
        0.0,
    )

    second_embedding = make_embedding(
        0.0,
        1.0,
    )

    first = tracker.update(
        [
            make_detection(
                100,
                50,
                140,
                150,
            )
        ],
        embeddings=np.array([first_embedding]),
    )

    assert first[0].track_id == 1

    second = tracker.update(
        [
            make_detection(
                102,
                51,
                142,
                151,
            )
        ],
        embeddings=np.array([second_embedding]),
    )

    assert len(second) == 1
    assert second[0].track_id == 1


def test_matching_cascade_prioritizes_recent_track():
    tracker = Tracker(
        TrackerConfig(
            min_iou=0.3,
            n_init=1,
            max_age=5,
            lambda_motion=0.98,
            max_cosine_distance=0.4415,
        )
    )

    embedding = make_embedding(
        1.0,
        0.0,
    )

    first = tracker.update(
        [
            make_detection(
                100,
                50,
                140,
                150,
            )
        ],
        embeddings=np.array([embedding]),
    )

    assert first[0].track_id == 1

    second = tracker.update(
        [
            make_detection(
                300,
                50,
                340,
                150,
            )
        ],
        embeddings=np.array([embedding]),
    )

    assert second[0].track_id == 2

    track_1 = next(track for track in tracker.tracks if track.track_id == 1)

    track_2 = next(track for track in tracker.tracks if track.track_id == 2)

    track_1.state[:4] = np.array(
        [
            120.0,
            100.0,
            40.0,
            100.0,
        ]
    )

    track_2.state[:4] = np.array(
        [
            120.0,
            100.0,
            40.0,
            100.0,
        ]
    )

    track_1.covariance *= 100.0
    track_2.covariance *= 100.0

    third = tracker.update(
        [
            make_detection(
                100,
                50,
                140,
                150,
            )
        ],
        embeddings=np.array([embedding]),
    )

    assert len(third) == 1
    assert third[0].track_id == 2


def test_low_confidence_detection_recovers_confirmed_track():
    tracker = Tracker(
        TrackerConfig(
            min_iou=0.3,
            n_init=1,
            max_age=30,
        )
    )

    first = tracker.update(
        [
            make_detection(
                100,
                50,
                140,
                150,
            )
        ]
    )

    assert len(first) == 1
    assert first[0].track_id == 1

    recovered = tracker.update(
        [],
        low_confidence_detections=[
            make_detection(
                102,
                51,
                142,
                151,
                confidence=0.2,
            )
        ],
    )

    assert len(recovered) == 1
    assert recovered[0].track_id == 1


def test_low_confidence_detection_does_not_start_track():
    tracker = Tracker(
        TrackerConfig(
            min_iou=0.3,
            n_init=1,
            max_age=30,
        )
    )

    tracks = tracker.update(
        [],
        low_confidence_detections=[
            make_detection(
                100,
                50,
                140,
                150,
                confidence=0.2,
            )
        ],
    )

    assert tracks == []
    assert tracker.tracks == []
