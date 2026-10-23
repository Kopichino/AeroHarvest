"""
Camera Vibration & Image Jitter Penalty Model (PRD Section 3.4).
Quantifies optical camera stability degradation from body rotation rates,
tilt angle exceedances, and temporal wind gust derivatives.
"""

from typing import Tuple, Union
import numpy as np
import torch


class CameraStabilityModel:
    """
    Mathematical Implementation of PRD Equation 178:
    C_camera = integral( alpha_1 * ||omega_body||^2 + alpha_2 * max(0, theta_tilt - theta_max)^2 + alpha_3 * ||d(v_wind)/dt||^2 ) dt
    """

    def __init__(
        self,
        alpha_1: float = 10.0,
        alpha_2: float = 50.0,
        alpha_3: float = 2.0,
        theta_max_nominal_deg: float = 15.0,
        theta_max_stable_deg: float = 12.0,
    ):
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.alpha_3 = alpha_3
        self.theta_max_nominal_rad = np.radians(theta_max_nominal_deg)
        self.theta_max_stable_rad = np.radians(theta_max_stable_deg)

    def evaluate_cost_step_np(
        self,
        omega_body: np.ndarray,
        theta_tilt_rad: float,
        wind_derivative: np.ndarray,
        is_stable_capture_mode: bool = False,
    ) -> float:
        """Computes instantaneous camera stability penalty rate (cost/s)."""
        theta_limit = self.theta_max_stable_rad if is_stable_capture_mode else self.theta_max_nominal_rad

        # 1. Body angular rate term
        term_omega = self.alpha_1 * float(np.sum(omega_body ** 2))

        # 2. Tilt angle exceedance penalty
        tilt_excess = max(0.0, theta_tilt_rad - theta_limit)
        term_tilt = self.alpha_2 * (tilt_excess ** 2)

        # 3. Wind gust derivative penalty
        term_gust = self.alpha_3 * float(np.sum(wind_derivative ** 2))

        return term_omega + term_tilt + term_gust

    def evaluate_cost_batch_torch(
        self,
        omega_body_tensor: torch.Tensor,
        tilt_angle_rad_tensor: torch.Tensor,
        wind_derivative_tensor: torch.Tensor,
        is_stable_capture_mode: bool = False,
    ) -> torch.Tensor:
        """
        Batch evaluation for MPPI trajectory rollouts.
        omega_body_tensor: (K, N, 3) or (K, 3)
        tilt_angle_rad_tensor: (K, N) or (K,)
        wind_derivative_tensor: (K, N, 3) or (K, 3)
        Returns: Tensor of same leading shape with camera penalty.
        """
        device = omega_body_tensor.device
        dtype = omega_body_tensor.dtype

        theta_limit = self.theta_max_stable_rad if is_stable_capture_mode else self.theta_max_nominal_rad
        theta_limit_t = torch.tensor(theta_limit, device=device, dtype=dtype)

        # Angular rate penalty
        term_omega = self.alpha_1 * torch.sum(omega_body_tensor ** 2, dim=-1)

        # Tilt exceedance penalty
        tilt_excess = torch.clamp(tilt_angle_rad_tensor - theta_limit_t, min=0.0)
        term_tilt = self.alpha_2 * (tilt_excess ** 2)

        # Gust derivative penalty
        term_gust = self.alpha_3 * torch.sum(wind_derivative_tensor ** 2, dim=-1)

        return term_omega + term_tilt + term_gust

    def compute_rms_jitter_deg(
        self,
        omega_history: np.ndarray,
        tilt_history: np.ndarray,
    ) -> float:
        """Computes aggregate RMS angular jitter in degrees for metric reporting."""
        if len(omega_history) == 0:
            return 0.0
        # High-frequency angular movement RMS
        omega_norms_deg = np.degrees(np.linalg.norm(omega_history, axis=1))
        tilt_excess_deg = np.maximum(0.0, np.degrees(tilt_history) - np.degrees(self.theta_max_nominal_rad))
        jitter_rms = float(np.sqrt(np.mean(omega_norms_deg ** 2) + 0.5 * np.mean(tilt_excess_deg ** 2)))
        return jitter_rms
