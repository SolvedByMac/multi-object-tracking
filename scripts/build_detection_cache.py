from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import torch

from src.data.mot_dataset import load_sequence_info
from src.detection.cache import (
    detections_to_array,
    save_detection_cache,
)
from src.detection.detector import PersonDetector

SEQUENCES = [
    "MOT17-02-FRCNN",
    "MOT17-04-FRCNN",
    "MOT17-05-FRCNN",
    "MOT17-09-FRCNN",
    "MOT17-10-FRCNN",
    "MOT17-11-FRCNN",
    "MOT17-13-FRCNN",
]


def build_sequence_cache(
    detector: PersonDetector,
    sequence_dir: Path,
    output_path: Path,
) -> tuple[int, float]:
    info = load_sequence_info(sequence_dir)

    frame_detections = {}

    start = time.perf_counter()

    for frame_idx in range(1, info.seq_length + 1):
        image_path = info.image_dir / f"{frame_idx:06d}.jpg"

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)

        detections = detector.detect(image)

        frame_detections[frame_idx] = detections_to_array(detections)

        if frame_idx % 100 == 0 or frame_idx == info.seq_length:
            print(f"{info.name}: {frame_idx}/{info.seq_length} frames")

    elapsed = time.perf_counter() - start

    save_detection_cache(output_path, frame_detections)

    return info.seq_length, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/MOT17/train"),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("cache/detections"),
    )

    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    detector = PersonDetector(device=device)

    total_frames = 0
    total_time = 0.0

    for sequence_name in SEQUENCES:
        sequence_dir = args.dataset_root / sequence_name
        output_path = args.output_root / f"{sequence_name}.npz"

        print()
        print(f"Building cache for {sequence_name}")

        frames, elapsed = build_sequence_cache(
            detector,
            sequence_dir,
            output_path,
        )

        fps = frames / elapsed

        print(f"{sequence_name}: {frames} frames in {elapsed:.1f}s ({fps:.2f} FPS)")

        total_frames += frames
        total_time += elapsed

    overall_fps = total_frames / total_time

    print()
    print("=== Detection cache complete ===")
    print(f"Frames: {total_frames}")
    print(f"Time: {total_time:.1f}s")
    print(f"Overall FPS: {overall_fps:.2f}")


if __name__ == "__main__":
    main()
