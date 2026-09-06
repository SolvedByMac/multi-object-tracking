from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.formats import write_mot_txt
from src.data.mot_dataset import get_split_frame_range, load_sequence_info
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

LAMBDA_VALUES = [
    0.00,
    0.30,
    0.50,
    0.70,
    0.98,
]

HIGH_CONFIDENCE = 0.10


def make_detections(rows: np.ndarray) -> list[Detection]:
    return [
        Detection(
            xyxy=row[:4].astype(float),
            confidence=float(row[4]),
        )
        for row in rows
    ]


def tracker_name(lambda_motion: float) -> str:
    return f"lambda_{lambda_motion:.2f}".replace(".", "_")


def run_sequence(
    sequence_name: str,
    lambda_motion: float,
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

    embedding_cache = load_embedding_cache(
        embedding_cache_root / f"{sequence_name}.npz"
    )

    tracker = Tracker(
        TrackerConfig(
            min_iou=0.3,
            n_init=3,
            max_age=30,
            lambda_motion=lambda_motion,
            max_cosine_distance=0.30,
            use_mahalanobis=True,
            use_appearance=True,
            use_low_confidence_recovery=False,
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

        high_mask = cached_detections[:, 4] >= HIGH_CONFIDENCE

        high_rows = cached_detections[high_mask]
        high_embeddings = cached_embeddings[high_mask]

        detections = make_detections(high_rows)

        tracks = tracker.update(
            detections=detections,
            embeddings=high_embeddings,
            frame_idx=frame_idx,
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

    name = tracker_name(lambda_motion)

    output_path = output_root / name / "data" / f"{sequence_name}.txt"

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_mot_txt(
        output_path,
        rows,
    )

    print(f"{name} {sequence_name}: {len(frame_range)} frames, {len(rows)} track rows")


def main() -> None:
    data_root = Path("data/MOT17/train")
    detection_cache_root = Path("cache/detections")
    embedding_cache_root = Path("cache/embeddings")
    output_root = Path("results/trackers/MOT17-val")

    for lambda_motion in LAMBDA_VALUES:
        print()
        print(f"lambda_motion={lambda_motion:.2f}")

        for sequence_name in SEQUENCES:
            run_sequence(
                sequence_name=sequence_name,
                lambda_motion=lambda_motion,
                data_root=data_root,
                detection_cache_root=detection_cache_root,
                embedding_cache_root=embedding_cache_root,
                output_root=output_root,
            )


if __name__ == "__main__":
    main()
