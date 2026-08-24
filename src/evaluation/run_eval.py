from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.formats import write_mot_txt
from src.data.mot_dataset import (
    filter_ground_truth_by_frames,
    half_split,
    load_ground_truth,
    load_sequence_info,
)

SEQUENCES = [
    "MOT17-02-FRCNN",
    "MOT17-04-FRCNN",
    "MOT17-05-FRCNN",
    "MOT17-09-FRCNN",
    "MOT17-10-FRCNN",
    "MOT17-11-FRCNN",
    "MOT17-13-FRCNN",
]


def gt_to_tracker_rows(gt: np.ndarray) -> np.ndarray:
    """
    Convert MOT17 GT rows to tracker-output format:

    frame, id, left, top, width, height, conf, x, y, z
    """
    rows = np.empty((len(gt), 10), dtype=float)

    rows[:, :6] = gt[:, :6]
    rows[:, 6] = 1.0
    rows[:, 7:] = -1.0

    return rows


def build_self_eval_files(
    dataset_root: str | Path = "data/MOT17/train",
    output_root: str | Path = "results/self_eval",
) -> None:
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    for sequence_name in SEQUENCES:
        sequence_dir = dataset_root / sequence_name

        info = load_sequence_info(sequence_dir)
        gt = load_ground_truth(sequence_dir)

        _, val_frames = half_split(info.seq_length)
        val_gt = filter_ground_truth_by_frames(gt, val_frames)

        tracker_rows = gt_to_tracker_rows(val_gt)

        output_path = output_root / f"{sequence_name}.txt"
        write_mot_txt(output_path, tracker_rows)

        print(f"{sequence_name}: {len(tracker_rows)} rows -> {output_path}")


if __name__ == "__main__":
    build_self_eval_files()
