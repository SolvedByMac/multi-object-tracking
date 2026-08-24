import numpy as np

from src.data.formats import (
    cxcywh_to_tlwh,
    read_mot_txt,
    tlwh_to_cxcywh,
    tlwh_to_xyxy,
    write_mot_txt,
    xyxy_to_tlwh,
)


def test_tlwh_xyxy_round_trip():
    original = np.array([100, 50, 40, 80], dtype=float)

    converted = tlwh_to_xyxy(original)
    restored = xyxy_to_tlwh(converted)

    np.testing.assert_allclose(restored, original)


def test_tlwh_cxcywh_round_trip():
    original = np.array([100, 50, 40, 80], dtype=float)

    converted = tlwh_to_cxcywh(original)
    restored = cxcywh_to_tlwh(converted)

    np.testing.assert_allclose(restored, original)


def test_mot_txt_round_trip(tmp_path):
    rows = np.array(
        [
            [1, 7, 100, 50, 40, 80, 1, -1, -1, -1],
            [2, 7, 102, 51, 40, 80, 1, -1, -1, -1],
        ],
        dtype=float,
    )

    path = tmp_path / "tracks.txt"

    write_mot_txt(path, rows)
    restored = read_mot_txt(path)

    np.testing.assert_allclose(restored, rows)
