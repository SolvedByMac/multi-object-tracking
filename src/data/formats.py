from __future__ import annotations

from pathlib import Path

import numpy as np


def tlwh_to_xyxy(box: np.ndarray) -> np.ndarray:
    left, top, width, height = np.asarray(box, dtype=float)
    return np.array([left, top, left + width, top + height], dtype=float)


def xyxy_to_tlwh(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(box, dtype=float)
    return np.array([x1, y1, x2 - x1, y2 - y1], dtype=float)


def tlwh_to_cxcywh(box: np.ndarray) -> np.ndarray:
    left, top, width, height = np.asarray(box, dtype=float)
    return np.array([left + width / 2.0, top + height / 2.0, width, height], dtype=float)


def cxcywh_to_tlwh(box: np.ndarray) -> np.ndarray:
    cx, cy, width, height = np.asarray(box, dtype=float)
    return np.array([cx - width / 2.0, cy - height / 2.0, width, height], dtype=float)


def read_mot_txt(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, 10), dtype=float)

    rows = np.loadtxt(path, delimiter=",", dtype=float)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    return rows


def write_mot_txt(path: str | Path, rows: np.ndarray | list[list[float]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    array = np.asarray(rows, dtype=float)
    if array.size == 0:
        path.write_text("")
        return

    if array.ndim == 1:
        array = array.reshape(1, -1)

    np.savetxt(path, array, delimiter=",", fmt="%.6f")
