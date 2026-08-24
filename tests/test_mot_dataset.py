import numpy as np
import pytest

from src.data.mot_dataset import (
    filter_ground_truth_by_frames,
    get_split_frame_range,
    half_split,
)


def test_half_split_even_sequence():
    train, val = half_split(600)

    assert train.start == 1
    assert train.stop == 301
    assert val.start == 301
    assert val.stop == 601


def test_half_split_odd_sequence():
    train, val = half_split(837)

    assert train.start == 1
    assert train.stop == 419
    assert val.start == 419
    assert val.stop == 838


def test_get_split_frame_range_rejects_unknown_split():
    with pytest.raises(ValueError):
        get_split_frame_range(600, "test")


def test_filter_ground_truth_by_frames():
    gt = np.array(
        [
            [1, 1, 10, 10, 20, 40, 1, 1, 1.0],
            [2, 1, 11, 10, 20, 40, 1, 1, 1.0],
            [3, 1, 12, 10, 20, 40, 1, 1, 1.0],
        ],
        dtype=float,
    )

    filtered = filter_ground_truth_by_frames(gt, range(2, 4))

    np.testing.assert_array_equal(filtered[:, 0], np.array([2.0, 3.0]))
