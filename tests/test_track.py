import numpy as np

from src.detection.detector import Detection
from src.tracking.kalman import KalmanFilter
from src.tracking.track import Track, TrackStatus


def make_detection(
    x1: float = 100.0,
    y1: float = 50.0,
    x2: float = 140.0,
    y2: float = 150.0,
) -> Detection:
    return Detection(
        xyxy=np.array([x1, y1, x2, y2], dtype=float),
        confidence=0.9,
    )


def test_new_track_is_tentative():
    kf = KalmanFilter()

    track = Track.from_detection(
        make_detection(),
        track_id=1,
        kf=kf,
        n_init=3,
    )

    assert track.status == TrackStatus.TENTATIVE
    assert track.hits == 1
    assert track.time_since_update == 0


def test_track_becomes_confirmed_after_n_init_hits():
    kf = KalmanFilter()
    detection = make_detection()

    track = Track.from_detection(
        detection,
        track_id=1,
        kf=kf,
        n_init=3,
    )

    track.predict(kf)
    track.update(detection, kf)

    assert track.status == TrackStatus.TENTATIVE

    track.predict(kf)
    track.update(detection, kf)

    assert track.status == TrackStatus.CONFIRMED
    assert track.hits == 3


def test_tentative_track_is_deleted_when_missed():
    kf = KalmanFilter()

    track = Track.from_detection(
        make_detection(),
        track_id=1,
        kf=kf,
        n_init=3,
    )

    track.predict(kf)
    track.mark_missed()

    assert track.status == TrackStatus.DELETED


def test_confirmed_track_survives_short_occlusion():
    kf = KalmanFilter()
    detection = make_detection()

    track = Track.from_detection(
        detection,
        track_id=1,
        kf=kf,
        n_init=1,
        max_age=3,
    )

    track.predict(kf)
    track.mark_missed()

    assert track.status == TrackStatus.CONFIRMED


def test_confirmed_track_deleted_after_max_age():
    kf = KalmanFilter()
    detection = make_detection()

    track = Track.from_detection(
        detection,
        track_id=1,
        kf=kf,
        n_init=1,
        max_age=2,
    )

    for _ in range(3):
        track.predict(kf)
        track.mark_missed()

    assert track.status == TrackStatus.DELETED


def test_track_box_round_trip_shape():
    kf = KalmanFilter()

    track = Track.from_detection(
        make_detection(),
        track_id=1,
        kf=kf,
    )

    np.testing.assert_allclose(
        track.to_xyxy(),
        np.array([100.0, 50.0, 140.0, 150.0]),
    )
