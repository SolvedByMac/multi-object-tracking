from __future__ import annotations

import csv
import subprocess
import sys
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

LAMBDA_VALUES = [
    0.98,
    0.90,
    0.75,
    0.50,
]

COSINE_THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.4415,
]


def run_sequence(
    sequence_name: str,
    tracker_name: str,
    lambda_motion: float,
    max_cosine_distance: float,
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
            max_cosine_distance=max_cosine_distance,
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

        detections = [
            Detection(
                xyxy=row[:4].astype(float),
                confidence=float(row[4]),
            )
            for row in cached_detections
        ]

        tracks = tracker.update(
            detections=detections,
            embeddings=cached_embeddings,
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

    output_path = output_root / tracker_name / "data" / f"{sequence_name}.txt"

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_mot_txt(
        output_path,
        rows,
    )


def run_tracker(
    tracker_name: str,
    lambda_motion: float,
    max_cosine_distance: float,
) -> None:
    data_root = Path("data/MOT17/train")
    detection_cache_root = Path("cache/detections")
    embedding_cache_root = Path("cache/embeddings")
    output_root = Path("results/trackers/MOT17-val")

    for sequence_name in SEQUENCES:
        run_sequence(
            sequence_name=sequence_name,
            tracker_name=tracker_name,
            lambda_motion=lambda_motion,
            max_cosine_distance=max_cosine_distance,
            data_root=data_root,
            detection_cache_root=detection_cache_root,
            embedding_cache_root=embedding_cache_root,
            output_root=output_root,
        )


def run_trackeval(
    tracker_name: str,
) -> None:
    command = [
        sys.executable,
        "-m",
        "scripts.evaluate_ablation",
        "--tracker",
        tracker_name,
    ]

    subprocess.run(
        command,
        check=True,
    )


def read_summary(
    tracker_name: str,
) -> dict[str, str]:
    summary_path = (
        Path("results/trackers/MOT17-val") / tracker_name / "pedestrian_summary.txt"
    )

    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    with summary_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        lines = [line.strip() for line in file if line.strip()]

    headers = lines[0].split()
    values = lines[1].split()

    return dict(
        zip(
            headers,
            values,
            strict=True,
        )
    )


def main() -> None:
    results: list[dict[str, float | str]] = []

    for lambda_motion in LAMBDA_VALUES:
        for max_cosine_distance in COSINE_THRESHOLDS:
            tracker_name = f"ablation_l{lambda_motion:.2f}_t{max_cosine_distance:.4f}"

            tracker_name = tracker_name.replace(
                ".",
                "p",
            )

            print()
            print(f"Running {tracker_name}")

            run_tracker(
                tracker_name=tracker_name,
                lambda_motion=lambda_motion,
                max_cosine_distance=max_cosine_distance,
            )

            run_trackeval(tracker_name)

            summary = read_summary(tracker_name)

            row = {
                "tracker": tracker_name,
                "lambda_motion": lambda_motion,
                "max_cosine_distance": max_cosine_distance,
                "HOTA": float(summary["HOTA"]),
                "MOTA": float(summary["MOTA"]),
                "IDF1": float(summary["IDF1"]),
                "IDSW": float(summary["IDSW"]),
                "Frag": float(summary["Frag"]),
            }

            results.append(row)

            print(
                f"HOTA={row['HOTA']:.3f} "
                f"MOTA={row['MOTA']:.3f} "
                f"IDF1={row['IDF1']:.3f} "
                f"IDSW={int(row['IDSW'])} "
                f"Frag={int(row['Frag'])}"
            )

    results.sort(
        key=lambda row: (
            -float(row["IDF1"]),
            float(row["IDSW"]),
        )
    )

    output_path = Path("outputs/p4-ablation-results.csv")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "tracker",
                "lambda_motion",
                "max_cosine_distance",
                "HOTA",
                "MOTA",
                "IDF1",
                "IDSW",
                "Frag",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print("=== Ranked by IDF1 ===")

    for row in results:
        print(
            f"{row['tracker']}: "
            f"IDF1={float(row['IDF1']):.3f} "
            f"IDSW={int(row['IDSW'])} "
            f"HOTA={float(row['HOTA']):.3f} "
            f"MOTA={float(row['MOTA']):.3f} "
            f"Frag={int(row['Frag'])}"
        )

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
