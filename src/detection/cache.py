from __future__ import annotations

from pathlib import Path

import numpy as np

from src.detection.detector import Detection


def detections_to_array(detections: list[Detection]) -> np.ndarray:
    """
    Convert a list of Detection objects to an (N, 5) array:

    [x1, y1, x2, y2, confidence]
    """
    if not detections:
        return np.empty((0, 5), dtype=np.float32)

    rows = [
        [
            *detection.xyxy.tolist(),
            detection.confidence,
        ]
        for detection in detections
    ]

    return np.asarray(rows, dtype=np.float32)


def save_detection_cache(
    path: str | Path,
    frame_detections: dict[int, np.ndarray],
) -> None:
    """
    Save per-frame detections to a compressed NumPy archive.

    Keys are frame numbers; values are arrays of shape (N, 5).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {
        f"frame_{frame_idx:06d}": detections
        for frame_idx, detections in frame_detections.items()
    }

    np.savez_compressed(path, **arrays)


def load_detection_cache(
    path: str | Path,
) -> dict[int, np.ndarray]:
    """
    Load a compressed detection cache.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path) as data:
        return {int(key.removeprefix("frame_")): data[key] for key in data.files}
