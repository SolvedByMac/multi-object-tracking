import numpy as np
import pytest

from src.reid.gallery import AppearanceGallery


def test_first_embedding_is_normalized():
    gallery = AppearanceGallery(alpha=0.9)

    embedding = np.array(
        [3.0, 4.0],
        dtype=np.float32,
    )

    stored = gallery.update(
        track_id=1,
        embedding=embedding,
    )

    np.testing.assert_allclose(
        stored,
        np.array([0.6, 0.8]),
        atol=1e-6,
    )

    assert np.isclose(
        np.linalg.norm(stored),
        1.0,
    )


def test_ema_update():
    gallery = AppearanceGallery(alpha=0.9)

    first = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    second = np.array(
        [0.0, 1.0],
        dtype=np.float32,
    )

    gallery.update(
        track_id=1,
        embedding=first,
    )

    updated = gallery.update(
        track_id=1,
        embedding=second,
    )

    expected = np.array(
        [0.9, 0.1],
        dtype=np.float32,
    )

    expected /= np.linalg.norm(expected)

    np.testing.assert_allclose(
        updated,
        expected,
        atol=1e-6,
    )

    assert np.isclose(
        np.linalg.norm(updated),
        1.0,
    )


def test_tracks_have_independent_embeddings():
    gallery = AppearanceGallery(alpha=0.9)

    gallery.update(
        1,
        np.array([1.0, 0.0]),
    )

    gallery.update(
        2,
        np.array([0.0, 1.0]),
    )

    np.testing.assert_allclose(
        gallery.get(1),
        np.array([1.0, 0.0]),
    )

    np.testing.assert_allclose(
        gallery.get(2),
        np.array([0.0, 1.0]),
    )


def test_remove_track():
    gallery = AppearanceGallery()

    gallery.update(
        7,
        np.array([1.0, 0.0]),
    )

    assert gallery.contains(7)

    gallery.remove(7)

    assert not gallery.contains(7)
    assert gallery.get(7) is None


def test_zero_embedding_is_rejected():
    gallery = AppearanceGallery()

    with pytest.raises(
        ValueError,
        match="nonzero norm",
    ):
        gallery.update(
            1,
            np.zeros(512, dtype=np.float32),
        )


def test_invalid_alpha_is_rejected():
    with pytest.raises(
        ValueError,
        match="alpha",
    ):
        AppearanceGallery(alpha=1.0)
