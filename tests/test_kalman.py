import numpy as np

from src.tracking.kalman import KalmanFilter


def test_kalman_filter_reduces_position_rmse():
    rng = np.random.default_rng(42)

    num_frames = 100

    true_x = 100.0
    true_y = 50.0

    velocity_x = 2.0
    velocity_y = 1.0

    width = 40.0
    height = 100.0

    truth = []
    measurements = []

    for frame_idx in range(num_frames):
        x = true_x + velocity_x * frame_idx
        y = true_y + velocity_y * frame_idx

        truth.append(
            np.array(
                [x, y, width, height],
                dtype=float,
            )
        )

        noisy_measurement = np.array(
            [
                x + rng.normal(0.0, 5.0),
                y + rng.normal(0.0, 5.0),
                width + rng.normal(0.0, 2.0),
                height + rng.normal(0.0, 2.0),
            ],
            dtype=float,
        )

        measurements.append(noisy_measurement)

    truth = np.asarray(truth)
    measurements = np.asarray(measurements)

    kf = KalmanFilter()

    mean, covariance = kf.initiate(measurements[0])

    estimates = [mean[:4].copy()]

    for measurement in measurements[1:]:
        mean, covariance = kf.predict(
            mean,
            covariance,
        )

        mean, covariance = kf.update(
            mean,
            covariance,
            measurement,
        )

        estimates.append(mean[:4].copy())

    estimates = np.asarray(estimates)

    raw_position_error = measurements[:, :2] - truth[:, :2]
    filtered_position_error = estimates[:, :2] - truth[:, :2]

    raw_rmse = np.sqrt(np.mean(raw_position_error**2))

    filtered_rmse = np.sqrt(np.mean(filtered_position_error**2))

    print(f"Raw RMSE: {raw_rmse:.3f}")
    print(f"Filtered RMSE: {filtered_rmse:.3f}")

    assert filtered_rmse < raw_rmse
