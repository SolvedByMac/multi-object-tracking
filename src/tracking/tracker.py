from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.detection.detector import Detection
from src.reid.gallery import AppearanceGallery
from src.tracking.association import (
    associate_fused_cascade,
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
        self.gallery = AppearanceGallery(alpha=0.9)
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

        track_boxes = self._track_boxes()
        detection_boxes = self._detection_boxes(detections)

        if embeddings is None:
            (
                matches,
                unmatched_track_indices,
                unmatched_detection_indices,
            ) = associate_iou(
                track_boxes,
                detection_boxes,
                min_iou=self.config.min_iou,
            )
        else:
            (
                matches,
                unmatched_track_indices,
                unmatched_detection_indices,
            ) = self._associate_with_appearance(
                track_boxes=track_boxes,
                detection_boxes=detection_boxes,
                detection_embeddings=embeddings,
            )

        for track_idx, detection_idx in matches:
            track = self.tracks[track_idx]

            track.update(
                detections[detection_idx],
                self.kf,
            )

            if embeddings is not None:
                self.gallery.update(
                    track.track_id,
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

    def _associate_with_appearance(
        self,
        track_boxes: np.ndarray,
        detection_boxes: np.ndarray,
        detection_embeddings: np.ndarray,
    ) -> tuple[
        list[tuple[int, int]],
        list[int],
        list[int],
    ]:
        num_tracks = len(self.tracks)
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

        confirmed_indices: list[int] = []
        confirmed_embeddings: list[np.ndarray] = []

        for track_idx, track in enumerate(self.tracks):
            if not track.is_confirmed():
                continue

            embedding = self.gallery.get(track.track_id)

            if embedding is None:
                continue

            confirmed_indices.append(track_idx)

            confirmed_embeddings.append(embedding)

        matches: list[tuple[int, int]] = []
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()

        if confirmed_indices:
            confirmed_boxes = track_boxes[confirmed_indices]

            confirmed_means = np.asarray(
                [self.tracks[idx].state for idx in confirmed_indices],
                dtype=float,
            )

            confirmed_covariances = np.asarray(
                [self.tracks[idx].covariance for idx in confirmed_indices],
                dtype=float,
            )

            confirmed_time_since_update = np.asarray(
                [self.tracks[idx].time_since_update for idx in confirmed_indices],
                dtype=int,
            )

            confirmed_embeddings_array = np.asarray(
                confirmed_embeddings,
                dtype=float,
            )

            (
                appearance_matches,
                _,
                _,
            ) = associate_fused_cascade(
                track_boxes=confirmed_boxes,
                track_means=confirmed_means,
                track_covariances=confirmed_covariances,
                track_embeddings=confirmed_embeddings_array,
                track_time_since_update=confirmed_time_since_update,
                detection_boxes=detection_boxes,
                detection_embeddings=detection_embeddings,
                kf=self.kf,
                max_age=self.config.max_age,
                lambda_motion=self.config.lambda_motion,
                max_cosine_distance=(self.config.max_cosine_distance),
            )

            for local_track_idx, detection_idx in appearance_matches:
                track_idx = confirmed_indices[local_track_idx]

                matches.append(
                    (
                        track_idx,
                        detection_idx,
                    )
                )

                matched_tracks.add(track_idx)

                matched_detections.add(detection_idx)

        fallback_track_indices = [
            track_idx
            for track_idx, track in enumerate(self.tracks)
            if (
                track_idx not in matched_tracks
                and (track.is_tentative() or track.time_since_update == 1)
            )
        ]

        remaining_detection_indices = [
            detection_idx
            for detection_idx in range(num_detections)
            if detection_idx not in matched_detections
        ]

        if fallback_track_indices and remaining_detection_indices:
            fallback_boxes = track_boxes[fallback_track_indices]

            remaining_detection_boxes = detection_boxes[remaining_detection_indices]

            (
                fallback_matches,
                _,
                _,
            ) = associate_iou(
                fallback_boxes,
                remaining_detection_boxes,
                min_iou=self.config.min_iou,
            )

            for (
                local_track_idx,
                local_detection_idx,
            ) in fallback_matches:
                track_idx = fallback_track_indices[local_track_idx]

                detection_idx = remaining_detection_indices[local_detection_idx]

                matches.append(
                    (
                        track_idx,
                        detection_idx,
                    )
                )

                matched_tracks.add(track_idx)

                matched_detections.add(detection_idx)

        unmatched_track_indices = [
            track_idx
            for track_idx in range(num_tracks)
            if track_idx not in matched_tracks
        ]

        unmatched_detection_indices = [
            detection_idx
            for detection_idx in range(num_detections)
            if detection_idx not in matched_detections
        ]

        return (
            matches,
            unmatched_track_indices,
            unmatched_detection_indices,
        )

    def _track_boxes(self) -> np.ndarray:
        if not self.tracks:
            return np.empty(
                (0, 4),
                dtype=float,
            )

        return np.asarray(
            [track.to_xyxy() for track in self.tracks],
            dtype=float,
        )

    @staticmethod
    def _detection_boxes(
        detections: list[Detection],
    ) -> np.ndarray:
        if not detections:
            return np.empty(
                (0, 4),
                dtype=float,
            )

        return np.asarray(
            [detection.xyxy for detection in detections],
            dtype=float,
        )

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
