from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data.formats import write_mot_txt
from src.data.mot_dataset import (
    get_split_frame_range,
    load_sequence_info,
)
from src.detection.cache import load_detection_cache
from src.detection.detector import Detection
from src.reid.cache import load_embedding_cache
from src.tracking.tracker import Tracker, TrackerConfig

SEQUENCES = [
    "MOT17-02-FRCNN",
    "MOT17-04-FRCNN",
    "MOT17-05-FRCNN",
    "MOT17-09-FRCNN",
    "MOT17-10-FRCNN",
    "MOT17-11-FRCNN",
    "MOT17-13-FRCNN",
]


@dataclass(frozen=True)
class AblationConfig:
    name: str
    use_mahalanobis: bool
    use_appearance: bool
    use_low_confidence_recovery: bool
    high_confidence: float
    low_confidence: float | None


CONFIGS = [
    AblationConfig(
        name="iou_only",
        use_mahalanobis=False,
        use_appearance=False,
        use_low_confidence_recovery=False,
        high_confidence=0.10,
        low_confidence=None,
    ),
    AblationConfig(
        name="mahalanobis",
        use_mahalanobis=True,
        use_appearance=False,
        use_low_confidence_recovery=False,
        high_confidence=0.10,
        low_confidence=None,
    ),
    AblationConfig(
        name="appearance",
        use_mahalanobis=True,
        use_appearance=True,
        use_low_confidence_recovery=False,
        high_confidence=0.10,
        low_confidence=None,
    ),
    AblationConfig(
        name="final",
        use_mahalanobis=True,
        use_appearance=True,
        use_low_confidence_recovery=True,
        high_confidence=0.50,
        low_confidence=0.10,
    ),
]


def make_detections(rows: np.ndarray) -> list[Detection]:
    return [
        Detection(
            xyxy=row[:4].astype(float),
            confidence=float(row[4]),
        )
        for row in rows
    ]


def run_sequence(
    sequence_name: str,
    config: AblationConfig,
    data_root: Path,
    detection_cache_root: Path,
    embedding_cache_root: Path,
    output_root: Path,
) -> None:
    sequence_dir = data_root / sequence_name
    sequence_info = load_sequence_info(sequence_dir)

    frame_range = get_split_frame_range(
        sequence_info.seq_length,
        split="val",
    )

    detection_cache = load_detection_cache(
        detection_cache_root / f"{sequence_name}.npz"
    )

    embedding_cache = None

    if config.use_appearance:
        embedding_cache = load_embedding_cache(
            embedding_cache_root / f"{sequence_name}.npz"
        )

    tracker = Tracker(
        TrackerConfig(
            min_iou=0.3,
            n_init=3,
            max_age=30,
            lambda_motion=0.50,
            max_cosine_distance=0.30,
            use_mahalanobis=config.use_mahalanobis,
            use_appearance=config.use_appearance,
            use_low_confidence_recovery=config.use_low_confidence_recovery,
        )
    )

    rows: list[list[float]] = []

    for frame_idx in frame_range:
        cached_detections = detection_cache.get(frame_idx)

        if cached_detections is None:
            cached_detections = np.empty(
                (0, 5),
                dtype=np.float32,
            )

        confidences = cached_detections[:, 4]
        high_mask = confidences >= config.high_confidence

        high_rows = cached_detections[high_mask]
        high_detections = make_detections(high_rows)

        high_embeddings = None

        if config.use_appearance:
            cached_embeddings = embedding_cache.get(frame_idx)

            if cached_embeddings is None:
                cached_embeddings = np.empty(
                    (0, 512),
                    dtype=np.float32,
                )

            if len(cached_detections) != len(cached_embeddings):
                raise ValueError(
                    f"{sequence_name} frame {frame_idx}: "
                    "detection and embedding counts do not match"
                )

            high_embeddings = cached_embeddings[high_mask]

        low_detections: list[Detection] = []

        if config.use_low_confidence_recovery and config.low_confidence is not None:
            low_mask = (confidences >= config.low_confidence) & (
                confidences < config.high_confidence
            )

            low_rows = cached_detections[low_mask]
            low_detections = make_detections(low_rows)

        tracks = tracker.update(
            detections=high_detections,
            embeddings=high_embeddings,
            frame_idx=frame_idx,
            low_confidence_detections=low_detections,
        )

        for track in tracks:
            left, top, width, height = track.to_tlwh()

            rows.append(
                [
                    frame_idx,
                    track.track_id,
                    left,
                    top,
                    width,
                    height,
                    1.0,
                    -1,
                    -1,
                    -1,
                ]
            )

    output_path = output_root / config.name / "data" / f"{sequence_name}.txt"

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_mot_txt(
        output_path,
        rows,
    )

    print(
        f"{config.name} | "
        f"{sequence_name}: "
        f"{len(frame_range)} frames, "
        f"{len(rows)} track rows"
    )


def run_config(
    config: AblationConfig,
    data_root: Path,
    detection_cache_root: Path,
    embedding_cache_root: Path,
    output_root: Path,
) -> None:
    print()
    print("=" * 80)
    print(f"Running: {config.name}")
    print(
        f"Mahalanobis={config.use_mahalanobis}, "
        f"appearance={config.use_appearance}, "
        f"low_conf_recovery={config.use_low_confidence_recovery}, "
        f"high_conf={config.high_confidence}, "
        f"low_conf={config.low_confidence}"
    )
    print("=" * 80)

    for sequence_name in SEQUENCES:
        run_sequence(
            sequence_name=sequence_name,
            config=config,
            data_root=data_root,
            detection_cache_root=detection_cache_root,
            embedding_cache_root=embedding_cache_root,
            output_root=output_root,
        )


def main() -> None:
    data_root = Path("data/MOT17/train")
    detection_cache_root = Path("cache/detections")
    embedding_cache_root = Path("cache/embeddings")
    output_root = Path("results/trackers/MOT17-val")

    for config in CONFIGS:
        run_config(
            config=config,
            data_root=data_root,
            detection_cache_root=detection_cache_root,
            embedding_cache_root=embedding_cache_root,
            output_root=output_root,
        )


if __name__ == "__main__":
    main()
