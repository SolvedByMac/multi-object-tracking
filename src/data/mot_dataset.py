from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SequenceInfo:
    name: str
    image_dir: Path
    gt_path: Path
    seq_length: int
    image_width: int
    image_height: int
    frame_rate: int


def half_split(seq_length: int) -> tuple[range, range]:
    split = seq_length // 2 + 1
    return range(1, split), range(split, seq_length + 1)


def get_split_frame_range(seq_length: int, split: str) -> range:
    train, val = half_split(seq_length)
    if split == "train":
        return train
    if split == "val":
        return val
    if split in {"full", "all"}:
        return range(1, seq_length + 1)
    raise ValueError(f"Unknown split: {split}")


def filter_ground_truth_by_frames(gt: np.ndarray, frames: range) -> np.ndarray:
    if gt.size == 0:
        return gt.copy()

    start = frames.start
    stop = frames.stop
    mask = (gt[:, 0] >= start) & (gt[:, 0] < stop)
    return gt[mask]


def load_ground_truth(sequence_dir: str | Path) -> np.ndarray:
    sequence_dir = Path(sequence_dir)
    gt_path = sequence_dir / "gt" / "gt.txt"
    if not gt_path.exists() or gt_path.stat().st_size == 0:
        return np.empty((0, 9), dtype=float)

    rows = np.loadtxt(gt_path, delimiter=",", dtype=float)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    return rows


def load_sequence_info(sequence_dir: str | Path) -> SequenceInfo:
    sequence_dir = Path(sequence_dir)
    info_path = sequence_dir / "seqinfo.ini"

    parser = configparser.ConfigParser()
    if not parser.read(info_path):
        raise FileNotFoundError(info_path)

    section = parser["Sequence"]
    image_dir = sequence_dir / section.get("imDir", "img1")

    return SequenceInfo(
        name=section.get("name", sequence_dir.name),
        image_dir=image_dir,
        gt_path=sequence_dir / "gt" / "gt.txt",
        seq_length=section.getint("seqLength"),
        image_width=section.getint("imWidth"),
        image_height=section.getint("imHeight"),
        frame_rate=section.getint("frameRate"),
    )
