from pathlib import Path

import numpy as np

from src.data.mot_dataset import (
    get_split_frame_range,
    load_ground_truth,
    load_sequence_info,
)
from src.tracking.kalman import KalmanFilter

SEQUENCES = [
    "MOT17-02-FRCNN",
    "MOT17-04-FRCNN",
    "MOT17-05-FRCNN",
    "MOT17-09-FRCNN",
    "MOT17-10-FRCNN",
    "MOT17-11-FRCNN",
    "MOT17-13-FRCNN",
]


def xywh_to_cxcywh(row: np.ndarray) -> np.ndarray:
    x, y, w, h = row
    return np.array(
        [
            x + w / 2,
            y + h / 2,
            w,
            h,
        ],
        dtype=float,
    )


def main() -> None:
    data_root = Path("data/MOT17/train")
    kalman_filter = KalmanFilter()

    nis_values: list[float] = []

    for sequence_name in SEQUENCES:
        sequence_dir = data_root / sequence_name
        sequence_info = load_sequence_info(sequence_dir)
        frame_range = get_split_frame_range(sequence_info.seq_length, "val")
        ground_truth = load_ground_truth(sequence_dir)

        rows_by_id: dict[int, list[np.ndarray]] = {}

        for row in ground_truth:
            frame_idx = int(row[0])
            track_id = int(row[1])

            if frame_idx not in frame_range:
                continue

            if int(row[6]) != 1:
                continue

            if int(row[7]) != 1:
                continue

            if float(row[8]) < 0.5:
                continue

            rows_by_id.setdefault(track_id, []).append(row)

        sequence_nis: list[float] = []

        for rows in rows_by_id.values():
            rows.sort(key=lambda row: int(row[0]))

            first = rows[0]
            measurement = xywh_to_cxcywh(first[2:6])

            mean, covariance = kalman_filter.initiate(measurement)
            previous_frame = int(first[0])

            for row in rows[1:]:
                frame_idx = int(row[0])

                if frame_idx != previous_frame + 1:
                    measurement = xywh_to_cxcywh(row[2:6])
                    mean, covariance = kalman_filter.initiate(measurement)
                    previous_frame = frame_idx
                    continue

                mean, covariance = kalman_filter.predict(mean, covariance)

                measurement = xywh_to_cxcywh(row[2:6])

                nis = kalman_filter.gating_distance(
                    mean,
                    covariance,
                    measurement[None, :],
                )[0]

                sequence_nis.append(float(nis))
                nis_values.append(float(nis))

                mean, covariance = kalman_filter.update(
                    mean,
                    covariance,
                    measurement,
                )

                previous_frame = frame_idx

        values = np.asarray(sequence_nis, dtype=float)

        print(
            f"{sequence_name}: "
            f"n={len(values)} "
            f"mean={values.mean():.3f} "
            f"median={np.median(values):.3f} "
            f"p95={np.percentile(values, 95):.3f}"
        )

    values = np.asarray(nis_values, dtype=float)

    print()
    print(f"Samples: {len(values)}")
    print(f"Mean NIS: {values.mean():.3f}")
    print(f"Median NIS: {np.median(values):.3f}")
    print(f"P95 NIS: {np.percentile(values, 95):.3f}")
    print(f"Fraction below 9.4877: {(values <= 9.4877).mean() * 100:.2f}%")


if __name__ == "__main__":
    main()
