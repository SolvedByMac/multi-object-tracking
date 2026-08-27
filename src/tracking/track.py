from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from src.detection.detector import Detection
from src.tracking.kalman import KalmanFilter


class TrackStatus(Enum):
    TENTATIVE = auto()
    CONFIRMED = auto()
    DELETED = auto()


@dataclass
class Track:
    track_id: int
    state: np.ndarray
    covariance: np.ndarray
    hits: int
    age: int
    time_since_update: int
    status: TrackStatus
    n_init: int = 3
    max_age: int = 30

    @classmethod
    def from_detection(
        cls,
        detection: Detection,
        track_id: int,
        kf: KalmanFilter,
        n_init: int = 3,
        max_age: int = 30,
    ) -> Track:
        measurement = detection_to_cxcywh(detection)

        mean, covariance = kf.initiate(measurement)

        status = TrackStatus.CONFIRMED if n_init <= 1 else TrackStatus.TENTATIVE

        return cls(
            track_id=track_id,
            state=mean,
            covariance=covariance,
            hits=1,
            age=1,
            time_since_update=0,
            status=status,
            n_init=n_init,
            max_age=max_age,
        )

    def predict(self, kf: KalmanFilter) -> None:

        self.state, self.covariance = kf.predict(
            self.state,
            self.covariance,
        )

        self.age += 1
        self.time_since_update += 1

    def update(
        self,
        detection: Detection,
        kf: KalmanFilter,
    ) -> None:

        measurement = detection_to_cxcywh(detection)

        self.state, self.covariance = kf.update(
            self.state,
            self.covariance,
            measurement,
        )

        self.hits += 1
        self.time_since_update = 0

        if self.status == TrackStatus.TENTATIVE and self.hits >= self.n_init:
            self.status = TrackStatus.CONFIRMED

    def mark_missed(self) -> None:

        if (
            self.status == TrackStatus.TENTATIVE
            or self.time_since_update > self.max_age
        ):
            self.status = TrackStatus.DELETED

    def is_tentative(self) -> bool:
        return self.status == TrackStatus.TENTATIVE

    def is_confirmed(self) -> bool:
        return self.status == TrackStatus.CONFIRMED

    def is_deleted(self) -> bool:
        return self.status == TrackStatus.DELETED

    def to_tlwh(self) -> np.ndarray:

        cx, cy, width, height = self.state[:4]

        return np.array(
            [
                cx - width / 2,
                cy - height / 2,
                width,
                height,
            ],
            dtype=float,
        )

    def to_xyxy(self) -> np.ndarray:

        left, top, width, height = self.to_tlwh()

        return np.array(
            [
                left,
                top,
                left + width,
                top + height,
            ],
            dtype=float,
        )


def detection_to_cxcywh(
    detection: Detection,
) -> np.ndarray:

    x1, y1, x2, y2 = detection.xyxy

    width = x2 - x1
    height = y2 - y1

    return np.array(
        [
            x1 + width / 2,
            y1 + height / 2,
            width,
            height,
        ],
        dtype=float,
    )
