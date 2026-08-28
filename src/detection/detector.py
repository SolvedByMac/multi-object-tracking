from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    """A single pedestrian detection."""

    xyxy: np.ndarray
    confidence: float

    @property
    def tlwh(self) -> np.ndarray:
        """Return box as [left, top, width, height]."""
        x1, y1, x2, y2 = self.xyxy
        return np.array(
            [x1, y1, x2 - x1, y2 - y1],
            dtype=float,
        )


class PersonDetector:
    """YOLOv8 pedestrian detector used to build the detection cache."""

    def __init__(
        self,
        weights: str = "yolov8m.pt",
        image_size: int = 1280,
        confidence_threshold: float = 0.1,
        nms_iou_threshold: float = 0.7,
        device: str | int | None = None,
    ) -> None:
        self.model = YOLO(weights)
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.device = device

    def detect(
        self,
        image: str | Path | np.ndarray,
    ) -> list[Detection]:

        results = self.model.predict(
            source=image,
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            iou=self.nms_iou_threshold,
            classes=[0],
            device=self.device,
            verbose=False,
        )

        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()

        return [
            Detection(
                xyxy=box.astype(float),
                confidence=float(confidence),
            )
            for box, confidence in zip(xyxy, confidences, strict=True)
        ]
