from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.detection.detector import Detection
from src.reid.gallery import AppearanceGallery
from src.tracking.association import (
    associate_fused,
    associate_iou,
)
from src.tracking.kalman import KalmanFilter
from src.tracking.track import Track


@dataclass
class TrackerConfig:
    min_iou: float = 0.3
    n_init: int = 3
    max_age: int = 30
    lambda_motion: float = 0.98
    max_cosine_distance: float = 0.4415


class Tracker:
    def __init__(
        self,
        config: TrackerConfig | None = None,
    ) -> None:
        self.config = config or TrackerConfig()

        self.kf = KalmanFilter()

        self.gallery = AppearanceGallery(
            alpha=0.9,
        )

        self.tracks: list[Track] = []
        self.next_track_id = 1

    def update(
        self,
        detections: list[Detection],
        embeddings: np.ndarray | None = None,
        frame_idx: int | None = None,
    ) -> list[Track]:
        del frame_idx

        if embeddings is not None and len(embeddings) != len(detections):
            raise ValueError("embeddings must align with detections")

        for track in self.tracks:
            track.predict(self.kf)

        track_boxes = np.asarray(
            [track.to_xyxy() for track in self.tracks],
            dtype=float,
        )

        if len(self.tracks) == 0:
            track_boxes = np.empty(
                (0, 4),
                dtype=float,
            )

        detection_boxes = np.asarray(
            [detection.xyxy for detection in detections],
            dtype=float,
        )

        if len(detections) == 0:
            detection_boxes = np.empty(
                (0, 4),
                dtype=float,
            )

        use_appearance = embeddings is not None and len(self.tracks) > 0

        if use_appearance:
            track_embeddings = []

            for track in self.tracks:
                embedding = self.gallery.get(track.track_id)

                if embedding is None:
                    use_appearance = False
                    break

                track_embeddings.append(embedding)

        if use_appearance:
            track_embeddings_array = np.asarray(
                track_embeddings,
                dtype=float,
            )

            track_means = np.asarray(
                [track.state for track in self.tracks],
                dtype=float,
            )

            track_covariances = np.asarray(
                [track.covariance for track in self.tracks],
                dtype=float,
            )

            (
                matches,
                unmatched_track_indices,
                unmatched_detection_indices,
            ) = associate_fused(
                track_boxes=track_boxes,
                track_means=track_means,
                track_covariances=track_covariances,
                track_embeddings=track_embeddings_array,
                detection_boxes=detection_boxes,
                detection_embeddings=embeddings,
                kf=self.kf,
                lambda_motion=self.config.lambda_motion,
                max_cosine_distance=(self.config.max_cosine_distance),
            )

        else:
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

            if embeddings is not None:
                self.gallery.update(
                    self.tracks[track_idx].track_id,
                    embeddings[detection_idx],
                )

        for track_idx in unmatched_track_indices:
            self.tracks[track_idx].mark_missed()

        for detection_idx in unmatched_detection_indices:
            embedding = embeddings[detection_idx] if embeddings is not None else None

            self._start_track(
                detections[detection_idx],
                embedding=embedding,
            )

        deleted_track_ids = {
            track.track_id for track in self.tracks if track.is_deleted()
        }

        for track_id in deleted_track_ids:
            self.gallery.remove(track_id)

        self.tracks = [track for track in self.tracks if not track.is_deleted()]

        return [
            track
            for track in self.tracks
            if (track.is_confirmed() and track.time_since_update == 0)
        ]

    def _start_track(
        self,
        detection: Detection,
        embedding: np.ndarray | None = None,
    ) -> None:
        track = Track.from_detection(
            detection=detection,
            track_id=self.next_track_id,
            kf=self.kf,
            n_init=self.config.n_init,
            max_age=self.config.max_age,
        )

        self.tracks.append(track)

        if embedding is not None:
            self.gallery.update(
                self.next_track_id,
                embedding,
            )

        self.next_track_id += 1
