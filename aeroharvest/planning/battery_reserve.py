"""
Adaptive Confidence-Gated Battery Reserve Gate (PRD Section 3.6).
Dynamically inflates the Return-To-Base (RTB) energy threshold based on
the integrated spatial predictive uncertainty along the return trajectory.
"""

from typing import Tuple, List
import numpy as np


class ConfidenceGatedBatteryReserve:
    """
    Mathematical Implementation of PRD Equation 204:
    E_reserve = E_nominal_return * ( 1.0 + beta * integral_path_RTB( sigma_w(p) dp ) )
    """

    def __init__(
        self,
        nominal_return_energy_wh: float = 14.0,
        beta_uncertainty_inflation: float = 0.08,
        cruise_speed_m_s: float = 8.0,
        average_power_w: float = 400.0,
    ):
        self.e_nominal = nominal_return_energy_wh
        self.beta = beta_uncertainty_inflation
        self.v_cruise = cruise_speed_m_s
        self.p_avg = average_power_w

    def compute_nominal_energy_wh(self, distance_m: float) -> float:
        """Calculates nominal return energy in Wh based on straight-line return distance."""
        flight_time_hours = (distance_m / self.v_cruise) / 3600.0
        e_nom = self.p_avg * flight_time_hours
        return max(self.e_nominal, e_nom)

    def evaluate_reserve_threshold(
        self,
        current_position: np.ndarray,
        launch_position: np.ndarray,
        gp_spatial_model,
        num_path_samples: int = 15,
    ) -> Tuple[float, float, bool]:
        """
        Evaluates dynamic E_reserve based on line-of-sight path uncertainty.

        Returns:
            e_reserve_wh: Dynamic reserve threshold in Watt-hours
            integrated_uncertainty: Line integral of sigma_w along return path
            is_inflated: True if uncertainty expanded the nominal safety margin
        """
        dist_to_home = float(np.linalg.norm(current_position - launch_position))
        nominal_energy = self.compute_nominal_energy_wh(dist_to_home)

        # Sample points along straight-line return route
        alphas = np.linspace(0.0, 1.0, num_path_samples)
        path_samples = np.array([
            (1.0 - a) * current_position + a * launch_position for a in alphas
        ])

        # Query predictive uncertainty sigma_w^2 along path
        _, path_variances = gp_spatial_model.predict(path_samples)
        path_sigmas = np.sqrt(np.maximum(0.0, path_variances))

        # Trapezoidal numerical integration of sigma_w(p) dp
        # Path segment length ds = dist_to_home / (num_path_samples - 1)
        ds = dist_to_home / max(1, (num_path_samples - 1))
        integrated_sigma = float(np.sum(path_sigmas) * ds)

        # Dimensionless scale: normalize by typical domain scale (500m)
        uncertainty_factor = (integrated_sigma / 500.0)

        e_reserve = nominal_energy * (1.0 + self.beta * uncertainty_factor)
        is_inflated = uncertainty_factor > 0.5

        return e_reserve, integrated_sigma, is_inflated

    def should_trigger_rtb(
        self,
        current_battery_wh: float,
        current_position: np.ndarray,
        launch_position: np.ndarray,
        gp_spatial_model,
    ) -> Tuple[bool, float]:
        """
        Returns (trigger_rtb_bool, current_reserve_threshold_wh).
        """
        e_reserve, _, _ = self.evaluate_reserve_threshold(
            current_position, launch_position, gp_spatial_model
        )
        trigger = current_battery_wh <= e_reserve
        return trigger, e_reserve
