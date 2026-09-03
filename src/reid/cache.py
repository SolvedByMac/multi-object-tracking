from __future__ import annotations

from pathlib import Path

import numpy as np


def save_embedding_cache(
    path: str | Path,
    frame_embeddings: dict[int, np.ndarray],
) -> None:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {
        f"frame_{frame_idx:06d}": embeddings
        for frame_idx, embeddings in frame_embeddings.items()
    }

    np.savez_compressed(path, **arrays)


def load_embedding_cache(
    path: str | Path,
) -> dict[int, np.ndarray]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path) as data:
        return {int(key.removeprefix("frame_")): data[key] for key in data.files}
