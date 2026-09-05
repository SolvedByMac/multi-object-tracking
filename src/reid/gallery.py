from __future__ import annotations

import numpy as np


class AppearanceGallery:
    def __init__(self, alpha: float = 0.9) -> None:
        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha must satisfy 0 <= alpha < 1")

        self.alpha = alpha
        self._embeddings: dict[int, np.ndarray] = {}

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        if embedding.ndim != 1:
            raise ValueError("embedding must be a 1D array")

        norm = np.linalg.norm(embedding)

        if norm == 0:
            raise ValueError("embedding must have nonzero norm")

        return embedding / norm

    def update(
        self,
        track_id: int,
        embedding: np.ndarray,
    ) -> np.ndarray:

        embedding = self._normalize(embedding)

        if track_id not in self._embeddings:
            updated = embedding
        else:
            previous = self._embeddings[track_id]

            updated = self.alpha * previous + (1.0 - self.alpha) * embedding

            updated = self._normalize(updated)

        self._embeddings[track_id] = updated

        return updated.copy()

    def get(
        self,
        track_id: int,
    ) -> np.ndarray | None:
        embedding = self._embeddings.get(track_id)

        if embedding is None:
            return None

        return embedding.copy()

    def remove(
        self,
        track_id: int,
    ) -> None:
        self._embeddings.pop(track_id, None)

    def contains(
        self,
        track_id: int,
    ) -> bool:
        return track_id in self._embeddings

    def __len__(self) -> int:
        return len(self._embeddings)
