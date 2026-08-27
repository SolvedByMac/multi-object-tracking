from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.detection.detector import Detection
from src.tracking.association import associate_iou
from src.tracking.kalman import KalmanFilter
from src.tracking.track import Track


@dataclass
class TrackerConfig:
    min_iou: float = 0.3
    n_init: int = 3
    max_age: int = 30


class Tracker:
    def __init__(
        self,
        config: TrackerConfig | None = None,
    ) -> None:
        self.config = config or TrackerConfig()
        self.kf = KalmanFilter()

        self.tracks: list[Track] = []
        self.next_track_id = 1

    def update(
        self,
        detections: list[Detection],
        embeddings: np.ndarray | None = None,
        frame_idx: int | None = None,
    ) -> list[Track]:

        del embeddings
        del frame_idx

        for track in self.tracks:
            track.predict(self.kf)

        track_boxes = np.asarray(
            [track.to_xyxy() for track in self.tracks],
            dtype=float,
        )

        if len(self.tracks) == 0:
            track_boxes = np.empty((0, 4), dtype=float)

        detection_boxes = np.asarray(
            [detection.xyxy for detection in detections],
            dtype=float,
        )

        if len(detections) == 0:
            detection_boxes = np.empty((0, 4), dtype=float)

        (
            matches,
            unmatched_track_indices,
            unmatched_detection_indices,
        ) = associate_iou(
            track_boxes,
            detection_boxes,
            min_iou=self.config.min_iou,
        )

        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                detections[detection_idx],
                self.kf,
            )

        for track_idx in unmatched_track_indices:
            self.tracks[track_idx].mark_missed()

        for detection_idx in unmatched_detection_indices:
            self._start_track(detections[detection_idx])

        self.tracks = [track for track in self.tracks if not track.is_deleted()]

        return [
            track
            for track in self.tracks
            if track.is_confirmed() and track.time_since_update == 0
        ]

    def _start_track(
        self,
        detection: Detection,
    ) -> None:
        track = Track.from_detection(
            detection=detection,
            track_id=self.next_track_id,
            kf=self.kf,
            n_init=self.config.n_init,
            max_age=self.config.max_age,
        )

        self.tracks.append(track)
        self.next_track_id += 1
