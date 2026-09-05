import numpy as np

from src.reid.encoder import OSNetEncoder


def test_clip_box_inside_frame():
    box = np.array([10.2, 20.4, 50.7, 100.1])

    result = OSNetEncoder._clip_box(
        box,
        frame_width=200,
        frame_height=150,
    )

    assert result == (10, 20, 51, 101)


def test_clip_box_to_frame_boundaries():
    box = np.array([-20.0, -10.0, 250.0, 180.0])

    result = OSNetEncoder._clip_box(
        box,
        frame_width=200,
        frame_height=150,
    )

    assert result == (0, 0, 200, 150)


def test_clip_box_rejects_invalid_box():
    box = np.array([50.0, 50.0, 40.0, 40.0])

    result = OSNetEncoder._clip_box(
        box,
        frame_width=200,
        frame_height=150,
    )

    assert result is None
