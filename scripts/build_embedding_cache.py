from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from src.data.mot_dataset import load_sequence_info
from src.detection.cache import load_detection_cache
from src.reid.cache import save_embedding_cache
from src.reid.encoder import OSNetEncoder

SEQUENCES = [
    "MOT17-02-FRCNN",
    "MOT17-04-FRCNN",
    "MOT17-05-FRCNN",
    "MOT17-09-FRCNN",
    "MOT17-10-FRCNN",
    "MOT17-11-FRCNN",
    "MOT17-13-FRCNN",
]

DEFAULT_WEIGHTS = Path(
    "weights/reid/"
    "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_"
    "b64_fb10_softmax_labelsmooth_flip_jitter.pth"
)


def build_sequence_cache(
    encoder: OSNetEncoder,
    sequence_dir: Path,
    detection_cache_path: Path,
    output_path: Path,
) -> tuple[int, int, float]:
    info = load_sequence_info(sequence_dir)
    detections_by_frame = load_detection_cache(detection_cache_path)

    frame_embeddings: dict[int, np.ndarray] = {}

    total_detections = 0
    start = time.perf_counter()

    for frame_idx in range(1, info.seq_length + 1):
        if frame_idx not in detections_by_frame:
            raise KeyError(f"Detection cache missing frame {frame_idx} for {info.name}")

        detections = detections_by_frame[frame_idx]

        if detections.ndim != 2 or detections.shape[1] != 5:
            raise ValueError(
                f"Unexpected detection shape for {info.name} "
                f"frame {frame_idx}: {detections.shape}"
            )

        image_path = info.image_dir / f"{frame_idx:06d}.jpg"

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(image_path)

        boxes = detections[:, :4]

        embeddings = encoder.encode(image, boxes)

        if embeddings.shape != (len(detections), 512):
            raise RuntimeError(
                f"Embedding alignment error for {info.name} "
                f"frame {frame_idx}: "
                f"{len(detections)} detections but "
                f"embedding shape {embeddings.shape}"
            )

        frame_embeddings[frame_idx] = embeddings
        total_detections += len(detections)

        if frame_idx % 100 == 0 or frame_idx == info.seq_length:
            print(
                f"{info.name}: "
                f"{frame_idx}/{info.seq_length} frames, "
                f"{total_detections} detections"
            )

    elapsed = time.perf_counter() - start

    save_embedding_cache(
        output_path,
        frame_embeddings,
    )

    return info.seq_length, total_detections, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/MOT17/train"),
    )

    parser.add_argument(
        "--detection-root",
        type=Path,
        default=Path("cache/detections"),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("cache/embeddings"),
    )

    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Weights: {args.weights}")

    encoder = OSNetEncoder(
        weights_path=args.weights,
        device=device,
        batch_size=args.batch_size,
    )

    total_frames = 0
    total_detections = 0
    total_time = 0.0

    for sequence_name in SEQUENCES:
        sequence_dir = args.dataset_root / sequence_name

        detection_cache_path = args.detection_root / f"{sequence_name}.npz"

        output_path = args.output_root / f"{sequence_name}.npz"

        print()
        print(f"Building embedding cache for {sequence_name}")

        frames, detections, elapsed = build_sequence_cache(
            encoder=encoder,
            sequence_dir=sequence_dir,
            detection_cache_path=detection_cache_path,
            output_path=output_path,
        )

        fps = frames / elapsed

        if elapsed > 0:
            detection_rate = detections / elapsed
        else:
            detection_rate = 0.0

        print(
            f"{sequence_name}: "
            f"{frames} frames, "
            f"{detections} detections in "
            f"{elapsed:.1f}s "
            f"({fps:.2f} FPS, "
            f"{detection_rate:.2f} crops/s)"
        )

        total_frames += frames
        total_detections += detections
        total_time += elapsed

    overall_fps = total_frames / total_time
    overall_crop_rate = total_detections / total_time

    print()
    print("=== Embedding cache complete ===")
    print(f"Frames: {total_frames}")
    print(f"Detections: {total_detections}")
    print(f"Time: {total_time:.1f}s")
    print(f"Overall FPS: {overall_fps:.2f}")
    print(f"Overall crop rate: {overall_crop_rate:.2f} crops/s")


if __name__ == "__main__":
    main()
