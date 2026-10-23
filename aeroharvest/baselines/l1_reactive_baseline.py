"""
Baseline 2: Reactive Disturbance Control (L1 Adaptive) (PRD Section 6.2).
Uses local disturbance observer to estimate instantaneous wind at current position
and applies reactive feedback compensation, but lacks spatial map and future look-ahead.
"""

from typing import Tuple
import numpy as np


class L1ReactiveBaseline:
    """
    Reactive disturbance rejection baseline.
    Cancels local disturbance wind, but has no spatial awareness or look-ahead.
    """

    def __init__(
        self,
        nominal_speed_m_s: float = 8.0,
        fixed_rtb_percentage: float = 16.0,
        l1_adaptation_gain: float = 0.6,
    ):
        self.v_nominal = nominal_speed_m_s
        self.rtb_thresh = fixed_rtb_percentage
        self.gain = l1_adaptation_gain

    def step(
        self,
        current_pos: np.ndarray,
        target_wp: np.ndarray,
        battery_percentage: float,
        launch_point: np.ndarray,
        local_wind_est: np.ndarray,
    ) -> Tuple[np.ndarray, float, str]:
        # Fixed battery return
        if battery_percentage <= self.rtb_thresh:
            diff = launch_point - current_pos
            dist = np.linalg.norm(diff)
            if dist < 5.0:
                return np.zeros(3), 0.0, "RETURN_TO_BASE_LANDED"
            v_dir = (diff / dist) * self.v_nominal
            # Reactive crab angle compensation against current local crosswind
            v_comp = v_dir - self.gain * local_wind_est
            v_comp = np.clip(v_comp, -12.0, 12.0)
            return v_comp, 0.0, "RETURN_TO_BASE_REACTIVE"

        diff = target_wp - current_pos
        dist = np.linalg.norm(diff)
        if dist < 1.0:
            return np.zeros(3), 0.0, "WAYPOINT_REACHED"

        unit_dir = diff / dist
        v_base = unit_dir * self.v_nominal

        # L1-style instantaneous disturbance compensation
        v_cmd = v_base - self.gain * local_wind_est
        v_cmd = np.clip(v_cmd, -12.0, 12.0)

        return v_cmd, 0.0, "REACTIVE_CRUISE"
