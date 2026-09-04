from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.data.mot_dataset import (
    get_split_frame_range,
    load_ground_truth,
    load_sequence_info,
)
from src.detection.cache import load_detection_cache
from src.reid.cache import load_embedding_cache
from src.tracking.association import iou

SEQUENCES = [
    "MOT17-02-FRCNN",
    "MOT17-04-FRCNN",
    "MOT17-05-FRCNN",
    "MOT17-09-FRCNN",
    "MOT17-10-FRCNN",
    "MOT17-11-FRCNN",
    "MOT17-13-FRCNN",
]


def tlwh_to_xyxy(box: np.ndarray) -> np.ndarray:
    x, y, width, height = box

    return np.array(
        [
            x,
            y,
            x + width,
            y + height,
        ],
        dtype=float,
    )


def match_detections_to_ground_truth(
    detections: np.ndarray,
    ground_truth: np.ndarray,
    min_iou: float,
) -> list[tuple[int, int]]:

    if len(detections) == 0 or len(ground_truth) == 0:
        return []

    cost_matrix = np.empty(
        (len(detections), len(ground_truth)),
        dtype=float,
    )

    detection_boxes = detections[:, :4]

    ground_truth_boxes = np.asarray(
        [tlwh_to_xyxy(row[2:6]) for row in ground_truth],
        dtype=float,
    )

    for detection_idx, detection_box in enumerate(detection_boxes):
        for gt_idx, gt_box in enumerate(ground_truth_boxes):
            cost_matrix[
                detection_idx,
                gt_idx,
            ] = 1.0 - iou(
                detection_box,
                gt_box,
            )

    detection_indices, gt_indices = linear_sum_assignment(cost_matrix)

    matches: list[tuple[int, int]] = []

    for detection_idx, gt_idx in zip(
        detection_indices,
        gt_indices,
        strict=True,
    ):
        overlap = (
            1.0
            - cost_matrix[
                detection_idx,
                gt_idx,
            ]
        )

        if overlap < min_iou:
            continue

        matches.append((detection_idx, gt_idx))

    return matches


def collect_identity_embeddings(
    dataset_root: Path,
    detection_cache_root: Path,
    embedding_cache_root: Path,
    min_iou: float,
) -> tuple[
    dict[int, list[np.ndarray]],
    list[list[tuple[int, np.ndarray]]],
]:

    identity_embeddings: dict[
        int,
        list[np.ndarray],
    ] = defaultdict(list)

    frame_groups: list[list[tuple[int, np.ndarray]]] = []

    identity_offset = 0

    for sequence_name in SEQUENCES:
        sequence_dir = dataset_root / sequence_name

        info = load_sequence_info(sequence_dir)

        ground_truth = load_ground_truth(sequence_dir)

        detections_by_frame = load_detection_cache(
            detection_cache_root / f"{sequence_name}.npz"
        )

        embeddings_by_frame = load_embedding_cache(
            embedding_cache_root / f"{sequence_name}.npz"
        )

        train_frames = get_split_frame_range(
            info.seq_length,
            "train",
        )

        matched_count = 0

        for frame_idx in train_frames:
            detections = detections_by_frame[frame_idx]

            embeddings = embeddings_by_frame[frame_idx]

            if len(detections) != len(embeddings):
                raise ValueError(
                    f"{sequence_name} frame "
                    f"{frame_idx}: detection/"
                    "embedding cache misalignment"
                )

            frame_gt = ground_truth[ground_truth[:, 0] == frame_idx]

            if len(frame_gt) == 0:
                continue

            # MOT17 pedestrian filtering:
            # mark == 1
            # class == 1
            # visibility >= 0.5
            valid = (
                (frame_gt[:, 6] == 1) & (frame_gt[:, 7] == 1) & (frame_gt[:, 8] >= 0.5)
            )

            frame_gt = frame_gt[valid]

            if len(frame_gt) == 0:
                continue

            matches = match_detections_to_ground_truth(
                detections,
                frame_gt,
                min_iou=min_iou,
            )

            current_frame: list[tuple[int, np.ndarray]] = []

            for detection_idx, gt_idx in matches:
                embedding = embeddings[detection_idx]

                if np.linalg.norm(embedding) == 0:
                    continue

                gt_id = int(frame_gt[gt_idx, 1])

                global_id = identity_offset + gt_id

                identity_embeddings[global_id].append(embedding)

                current_frame.append(
                    (
                        global_id,
                        embedding,
                    )
                )

                matched_count += 1

            if len(current_frame) >= 2:
                frame_groups.append(current_frame)

        max_sequence_id = int(np.max(ground_truth[:, 1])) if len(ground_truth) else 0

        identity_offset += max_sequence_id + 1000

        print(f"{sequence_name}: {matched_count} matched crops")

    return (
        identity_embeddings,
        frame_groups,
    )


def cosine_distance(
    embedding_a: np.ndarray,
    embedding_b: np.ndarray,
) -> float:
    similarity = float(
        np.dot(
            embedding_a,
            embedding_b,
        )
    )

    similarity = float(
        np.clip(
            similarity,
            -1.0,
            1.0,
        )
    )

    return 1.0 - similarity


def sample_same_identity_distances(
    identity_embeddings: dict[
        int,
        list[np.ndarray],
    ],
    max_pairs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    distances: list[float] = []

    identities = [
        identity
        for identity, embeddings in identity_embeddings.items()
        if len(embeddings) >= 2
    ]

    while len(distances) < max_pairs and identities:
        identity = int(rng.choice(identities))

        embeddings = identity_embeddings[identity]

        indices = rng.choice(
            len(embeddings),
            size=2,
            replace=False,
        )

        embedding_a = embeddings[int(indices[0])]

        embedding_b = embeddings[int(indices[1])]

        distances.append(
            cosine_distance(
                embedding_a,
                embedding_b,
            )
        )

    return np.asarray(
        distances,
        dtype=float,
    )


def sample_different_identity_distances(
    frame_groups: list[list[tuple[int, np.ndarray]]],
    max_pairs: int,
    rng: np.random.Generator,
) -> np.ndarray:

    eligible_frames = [
        frame for frame in frame_groups if len({identity for identity, _ in frame}) >= 2
    ]

    distances: list[float] = []

    while len(distances) < max_pairs and eligible_frames:
        frame = eligible_frames[int(rng.integers(len(eligible_frames)))]

        first_idx, second_idx = rng.choice(
            len(frame),
            size=2,
            replace=False,
        )

        identity_a, embedding_a = frame[int(first_idx)]

        identity_b, embedding_b = frame[int(second_idx)]

        if identity_a == identity_b:
            continue

        distances.append(
            cosine_distance(
                embedding_a,
                embedding_b,
            )
        )

    return np.asarray(
        distances,
        dtype=float,
    )


def choose_threshold(
    same_distances: np.ndarray,
    different_distances: np.ndarray,
) -> tuple[float, float, float]:
    thresholds = np.linspace(
        0.0,
        1.0,
        2001,
    )

    best_threshold = 0.0
    best_balanced_error = float("inf")
    best_same_reject_rate = 0.0
    best_different_accept_rate = 0.0

    for threshold in thresholds:
        same_reject_rate = float(np.mean(same_distances > threshold))

        different_accept_rate = float(np.mean(different_distances <= threshold))

        balanced_error = 0.5 * (same_reject_rate + different_accept_rate)

        if balanced_error < best_balanced_error:
            best_balanced_error = balanced_error

            best_threshold = float(threshold)

            best_same_reject_rate = same_reject_rate

            best_different_accept_rate = different_accept_rate

    return (
        best_threshold,
        best_same_reject_rate,
        best_different_accept_rate,
    )


def print_threshold_stats(
    threshold: float,
    same_distances: np.ndarray,
    different_distances: np.ndarray,
) -> None:
    same_reject_rate = float(np.mean(same_distances > threshold))

    different_accept_rate = float(np.mean(different_distances <= threshold))

    balanced_error = 0.5 * (same_reject_rate + different_accept_rate)

    print(
        f"Threshold {threshold:.4f}: "
        f"same-ID rejected "
        f"{100 * same_reject_rate:.2f}% | "
        f"different-ID accepted "
        f"{100 * different_accept_rate:.2f}% | "
        f"balanced error "
        f"{100 * balanced_error:.2f}%"
    )


def save_distribution_plot(
    same_distances: np.ndarray,
    different_distances: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(10, 6))

    plt.hist(
        same_distances,
        bins=60,
        density=True,
        alpha=0.6,
        label="Same ID",
    )

    plt.hist(
        different_distances,
        bins=60,
        density=True,
        alpha=0.6,
        label="Different ID",
    )

    plt.axvline(
        threshold,
        linestyle="--",
        linewidth=2,
        label=(f"Selected threshold {threshold:.3f}"),
    )

    plt.axvline(
        0.25,
        linestyle=":",
        linewidth=2,
        label="Reference threshold 0.25",
    )

    plt.xlabel("Cosine distance")

    plt.ylabel("Density")

    plt.title("OSNet Re-ID cosine-distance distributions")

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=180,
    )

    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/MOT17/train"),
    )

    parser.add_argument(
        "--detection-cache-root",
        type=Path,
        default=Path("cache/detections"),
    )

    parser.add_argument(
        "--embedding-cache-root",
        type=Path,
        default=Path("cache/embeddings"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/p4-reid-distance-distributions.png"),
    )

    parser.add_argument(
        "--pairs",
        type=int,
        default=20000,
    )

    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    (
        identity_embeddings,
        frame_groups,
    ) = collect_identity_embeddings(
        dataset_root=args.dataset_root,
        detection_cache_root=(args.detection_cache_root),
        embedding_cache_root=(args.embedding_cache_root),
        min_iou=args.match_iou,
    )

    same_distances = sample_same_identity_distances(
        identity_embeddings,
        max_pairs=args.pairs,
        rng=rng,
    )

    different_distances = sample_different_identity_distances(
        frame_groups,
        max_pairs=args.pairs,
        rng=rng,
    )

    if len(same_distances) == 0:
        raise RuntimeError("No same-ID pairs were generated")

    if len(different_distances) == 0:
        raise RuntimeError("No different-ID pairs were generated")

    (
        threshold,
        same_reject_rate,
        different_accept_rate,
    ) = choose_threshold(
        same_distances,
        different_distances,
    )

    print()
    print("=== Re-ID distance distribution ===")

    print(f"Same-ID pairs:       {len(same_distances)}")

    print(f"Different-ID pairs:  {len(different_distances)}")

    print()

    print("Same-ID distance:")

    print(f"  mean:   {np.mean(same_distances):.4f}")

    print(f"  median: {np.median(same_distances):.4f}")

    print(f"  p95:    {np.percentile(same_distances, 95):.4f}")

    print()

    print("Different-ID distance:")

    print(f"  mean:   {np.mean(different_distances):.4f}")

    print(f"  median: {np.median(different_distances):.4f}")

    print(f"  p05:    {np.percentile(different_distances, 5):.4f}")

    print()

    print(f"Selected threshold: {threshold:.4f}")

    print(f"Same-ID rejection: {100 * same_reject_rate:.2f}%")

    print(f"Different-ID acceptance: {100 * different_accept_rate:.2f}%")

    print()

    print("Comparison:")

    print_threshold_stats(
        0.25,
        same_distances,
        different_distances,
    )

    print_threshold_stats(
        threshold,
        same_distances,
        different_distances,
    )

    save_distribution_plot(
        same_distances,
        different_distances,
        threshold,
        args.output,
    )

    print()
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()
