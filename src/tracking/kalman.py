from __future__ import annotations

import numpy as np


class KalmanFilter:
    def __init__(
        self,
        std_weight_position: float = 1.0 / 20,
        std_weight_velocity: float = 1.0 / 160,
    ) -> None:
        self.std_weight_position = std_weight_position
        self.std_weight_velocity = std_weight_velocity

        self.ndim = 4
        self.dt = 1.0

        # State transition matrix.
        self.motion_matrix = np.eye(2 * self.ndim, dtype=float)

        for i in range(self.ndim):
            self.motion_matrix[i, self.ndim + i] = self.dt

        # Measurement matrix: observe x, y, w, h only.
        self.measurement_matrix = np.eye(
            self.ndim,
            2 * self.ndim,
            dtype=float,
        )

    def initiate(
        self,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        measurement = np.asarray(measurement, dtype=float)

        if measurement.shape != (4,):
            raise ValueError(
                f"Expected measurement shape (4,), got {measurement.shape}"
            )

        mean = np.zeros(8, dtype=float)
        mean[:4] = measurement

        h = max(float(measurement[3]), 1.0)

        position_std = self.std_weight_position * h
        velocity_std = self.std_weight_velocity * h

        std = np.array(
            [
                2 * position_std,
                2 * position_std,
                2 * position_std,
                2 * position_std,
                10 * velocity_std,
                10 * velocity_std,
                10 * velocity_std,
                10 * velocity_std,
            ],
            dtype=float,
        )

        covariance = np.diag(std**2)

        return mean, covariance

    def predict(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        h = max(float(mean[3]), 1.0)

        position_std = self.std_weight_position * h
        velocity_std = self.std_weight_velocity * h

        motion_std = np.array(
            [
                position_std,
                position_std,
                position_std,
                position_std,
                velocity_std,
                velocity_std,
                velocity_std,
                velocity_std,
            ],
            dtype=float,
        )

        motion_covariance = np.diag(motion_std**2)

        predicted_mean = self.motion_matrix @ mean

        predicted_covariance = (
            self.motion_matrix @ covariance @ self.motion_matrix.T + motion_covariance
        )

        return predicted_mean, predicted_covariance

    def project(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        h = max(float(mean[3]), 1.0)

        measurement_std = self.std_weight_position * h

        measurement_covariance = np.diag(
            np.full(
                4,
                measurement_std**2,
                dtype=float,
            )
        )

        projected_mean = self.measurement_matrix @ mean

        projected_covariance = (
            self.measurement_matrix @ covariance @ self.measurement_matrix.T
            + measurement_covariance
        )

        return projected_mean, projected_covariance

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        measurement = np.asarray(measurement, dtype=float)

        projected_mean, projected_covariance = self.project(
            mean,
            covariance,
        )

        cross_covariance = covariance @ self.measurement_matrix.T

        kalman_gain = np.linalg.solve(
            projected_covariance,
            cross_covariance.T,
        ).T

        innovation = measurement - projected_mean

        updated_mean = mean + kalman_gain @ innovation

        updated_covariance = (
            covariance - kalman_gain @ projected_covariance @ kalman_gain.T
        )

        return updated_mean, updated_covariance
