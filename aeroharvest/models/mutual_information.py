"""
Mutual Information & Informative Path Planning Module (PRD Section 3.5).
Calculates the mutual information gain I(w_*; y) for candidate trajectory detours
to support Active Informative Path Planning (ACTIVE_PROBE mode).
"""

from typing import Union
import numpy as np
import torch


class MutualInformationMetric:
    """
    Mathematical Implementation of PRD Equation 194:
    I(w_*; y) = 0.5 * ln det( I + sigma_n^(-2) * K_** )
    Supports exact log-determinant or fast Cholesky/entropy bounds.
    """

    def __init__(
        self,
        lengthscale: float = 40.0,
        signal_variance: float = 2.0,
        observation_noise: float = 0.04,
    ):
        self.lengthscale = lengthscale
        self.sigma_f2 = signal_variance
        self.sigma_n2 = observation_noise
        self.inv_noise = 1.0 / self.sigma_n2

    def matern_52_kernel(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Computes Matern 5/2 covariance matrix between point sets x1 and x2."""
        dists = np.linalg.norm(x1[:, np.newaxis, :] - x2[np.newaxis, :, :], axis=-1)
        scaled_d = np.sqrt(5.0) * dists / self.lengthscale
        cov = self.sigma_f2 * (1.0 + scaled_d + (5.0 * dists ** 2) / (3.0 * self.lengthscale ** 2)) * np.exp(-scaled_d)
        return cov

    def compute_mutual_information(self, path_points: np.ndarray) -> float:
        """
        Computes mutual information gain I(w_*; y) along a set of query waypoints.
        path_points: (N, 3) spatial coordinates.
        """
        pts = np.asarray(path_points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)

        N = pts.shape[0]
        if N == 0:
            return 0.0

        # Subsample if path has too many points to keep latency well under 15ms
        if N > 24:
            indices = np.linspace(0, N - 1, 24, dtype=int)
            pts = pts[indices]
            N = 24

        K_star = self.matern_52_kernel(pts, pts)
        matrix = np.eye(N) + self.inv_noise * K_star

        # Robust log-determinant via SVD or slogdet
        sign, logdet = np.linalg.slogdet(matrix)
        if sign <= 0:
            # Fallback to diagonal entropy approximation
            diag_gain = 0.5 * np.sum(np.log(1.0 + self.inv_noise * np.diag(K_star)))
            return float(max(0.0, diag_gain))

        info_gain = 0.5 * logdet
        return float(max(0.0, info_gain))

    def compute_info_efficiency(
        self,
        candidate_path: np.ndarray,
        delta_energy_wh: float,
    ) -> float:
        """
        Computes mutual information efficiency metric:
        eta = I(w_*; y) / Delta E
        Used in ACTIVE_PROBE triggering condition (PRD Section 4.3).
        """
        info_gain = self.compute_mutual_information(candidate_path)
        denom = max(delta_energy_wh, 0.05)
        return info_gain / denom
