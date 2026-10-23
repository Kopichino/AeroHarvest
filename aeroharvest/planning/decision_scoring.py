"""
Unified Tactical Decision Scoring Engine (PRD Section 3.5).
Evaluates candidate high-level tactical waypoint actions and local paths
trading off target progress, mutual information, electrical power,
wind hazard risk, camera vibration jitter, and time delay.
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np


@dataclass
class ActionCandidate:
    action_name: str
    target_waypoint: np.ndarray
    candidate_path: np.ndarray
    expected_duration_s: float
    is_inspection_target: bool = False


class TacticalDecisionScorer:
    """
    Mathematical Implementation of PRD Equation 188:
    Score(a_k) = w1*Progress + w2*I(w_*;y) - w3*P_elec - w4*R_wind - w5*C_camera - w6*Delta_T
    """

    def __init__(
        self,
        w1_progress: float = 12.0,
        w2_info_gain: float = 8.0,
        w3_power: float = 0.03,
        w4_wind_hazard: float = 10.0,
        w5_camera: float = 6.0,
        w6_time: float = 1.5,
        wind_limit_m_s: float = 12.0,
    ):
        self.w1 = w1_progress
        self.w2 = w2_info_gain
        self.w3 = w3_power
        self.w4 = w4_wind_hazard
        self.w5 = w5_camera
        self.w6 = w6_time
        self.v_limit = wind_limit_m_s

    def compute_wind_hazard_risk(self, mean_wind_vec: np.ndarray, uncertainty_var: float) -> float:
        """
        PRD Equation 198:
        R_wind = max(0, ||mu_w|| + 2 * sigma_w - v_limit)
        """
        mu_norm = float(np.linalg.norm(mean_wind_vec))
        sigma_w = float(np.sqrt(max(0.0, uncertainty_var)))
        risk = max(0.0, mu_norm + 2.0 * sigma_w - self.v_limit)
        return risk

    def evaluate_action_score(
        self,
        current_pos: np.ndarray,
        mission_goal_pos: np.ndarray,
        action: ActionCandidate,
        power_model,
        camera_model,
        info_metric,
        gp_spatial_model,
        is_stable_capture: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluates the composite score and returns component breakdown.
        """
        # 1. Target Progress toward primary mission waypoint
        dist_before = float(np.linalg.norm(current_pos - mission_goal_pos))
        dist_after = float(np.linalg.norm(action.target_waypoint - mission_goal_pos))
        progress_m = max(0.0, dist_before - dist_after)

        # 2. Mutual Information Gain along action path
        info_gain = info_metric.compute_mutual_information(action.candidate_path)

        # 3. Electrical Power along candidate path
        # Sample midpoint for wind estimation
        midpoint = 0.5 * (current_pos + action.target_waypoint)
        mid_wind_mean, mid_wind_var = gp_spatial_model.predict(midpoint)
        mid_wind_mean = mid_wind_mean.ravel()

        nominal_v_ground = (action.target_waypoint - current_pos) / max(1e-2, action.expected_duration_s)
        v_rel_est = nominal_v_ground - mid_wind_mean
        p_elec = power_model.compute_power_np(v_rel_est)

        # 4. Wind Hazard Risk
        r_wind = self.compute_wind_hazard_risk(mid_wind_mean, float(mid_wind_var[0]))

        # 5. Camera Jitter Cost
        # Approximate body tilt under this relative wind
        lean_angle_est = np.arctan2(np.linalg.norm(nominal_v_ground[:2]), 9.81)
        c_camera = camera_model.evaluate_cost_step_np(
            omega_body=np.array([0.05, 0.05, 0.0]),
            theta_tilt_rad=float(lean_angle_est),
            wind_derivative=np.zeros(3),
            is_stable_capture_mode=is_stable_capture,
        )

        # 6. Time Delay
        delta_t = action.expected_duration_s

        composite_score = (
            self.w1 * (progress_m / 10.0)
            + self.w2 * info_gain
            - self.w3 * (p_elec / 100.0)
            - self.w4 * r_wind
            - self.w5 * c_camera
            - self.w6 * (delta_t / 5.0)
        )

        return {
            "score": float(composite_score),
            "progress": float(progress_m),
            "info_gain": float(info_gain),
            "power_w": float(p_elec),
            "wind_hazard": float(r_wind),
            "camera_cost": float(c_camera),
            "delta_t": float(delta_t),
        }
