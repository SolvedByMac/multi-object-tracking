import numpy as np

from src.reid.cache import (
    load_embedding_cache,
    save_embedding_cache,
)


def test_embedding_cache_round_trip(tmp_path):
    path = tmp_path / "embeddings.npz"

    expected = {
        1: np.random.default_rng(0).normal(size=(3, 512)).astype(np.float32),
        2: np.empty((0, 512), dtype=np.float32),
        3: np.random.default_rng(1).normal(size=(2, 512)).astype(np.float32),
    }

    save_embedding_cache(path, expected)
    actual = load_embedding_cache(path)

    assert actual.keys() == expected.keys()

    for frame_idx, expected_embeddings in expected.items():
        np.testing.assert_array_equal(
            actual[frame_idx],
            expected_embeddings,
        )


def test_embedding_cache_missing_file(tmp_path):
    path = tmp_path / "missing.npz"

    try:
        load_embedding_cache(path)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError")
