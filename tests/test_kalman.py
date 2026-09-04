import numpy as np
import pytest

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


def test_gating_distance_is_zero_at_prediction():
    kf = KalmanFilter()

    measurement = np.array(
        [100.0, 200.0, 50.0, 120.0],
        dtype=float,
    )

    mean, covariance = kf.initiate(measurement)

    projected_mean, _ = kf.project(
        mean,
        covariance,
    )

    distances = kf.gating_distance(
        mean,
        covariance,
        projected_mean.reshape(1, 4),
    )

    assert distances.shape == (1,)
    assert np.isclose(distances[0], 0.0)


def test_gating_distance_increases_for_far_measurement():
    kf = KalmanFilter()

    measurement = np.array(
        [100.0, 200.0, 50.0, 120.0],
        dtype=float,
    )

    mean, covariance = kf.initiate(measurement)

    candidates = np.array(
        [
            [101.0, 201.0, 50.0, 120.0],
            [300.0, 400.0, 50.0, 120.0],
        ],
        dtype=float,
    )

    distances = kf.gating_distance(
        mean,
        covariance,
        candidates,
    )

    assert distances[0] < distances[1]


def test_gating_distance_empty_measurements():
    kf = KalmanFilter()

    measurement = np.array(
        [100.0, 200.0, 50.0, 120.0],
        dtype=float,
    )

    mean, covariance = kf.initiate(measurement)

    distances = kf.gating_distance(
        mean,
        covariance,
        np.empty((0, 4)),
    )

    assert distances.shape == (0,)


def test_gating_distance_rejects_invalid_shape():
    kf = KalmanFilter()

    measurement = np.array(
        [100.0, 200.0, 50.0, 120.0],
        dtype=float,
    )

    mean, covariance = kf.initiate(measurement)

    with pytest.raises(
        ValueError,
        match="measurements must have shape",
    ):
        kf.gating_distance(
            mean,
            covariance,
            np.zeros((2, 3)),
        )


def test_mahalanobis_gate_threshold():
    from src.tracking.association import CHI2_95_4DOF

    assert np.isclose(
        CHI2_95_4DOF,
        9.4877,
    )
