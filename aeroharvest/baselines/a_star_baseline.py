"""
Baseline 1: Geometric Shortest Path (A*) Controller (PRD Section 6.2).
Flies straight geometric lines connecting waypoints, completely ignoring wind fields,
using a fixed 15% static battery return threshold and standard PX4 position control.
"""

from typing import Tuple
import numpy as np


class AStarGeometricBaseline:
    """
    Standard geometric shortest-path planner.
    Ignores spatial wind, causing high tilt angles and camera vibration in headwind.
    """

    def __init__(
        self,
        nominal_speed_m_s: float = 8.0,
        fixed_rtb_percentage: float = 15.0,
    ):
        self.v_nominal = nominal_speed_m_s
        self.rtb_thresh = fixed_rtb_percentage

    def step(
        self,
        current_pos: np.ndarray,
        target_wp: np.ndarray,
        battery_percentage: float,
        launch_point: np.ndarray,
    ) -> Tuple[np.ndarray, float, str]:
        """
        Computes velocity command and yaw rate command.
        """
        # Fixed battery return failsafe (static 15%)
        if battery_percentage <= self.rtb_thresh:
            direction = launch_point - current_pos
            dist = np.linalg.norm(direction)
            if dist < 5.0:
                return np.zeros(3), 0.0, "RETURN_TO_BASE_LANDED"
            v_cmd = (direction / dist) * self.v_nominal
            return v_cmd, 0.0, "RETURN_TO_BASE_FIXED"

        # Direct geometric vector toward target waypoint
        diff = target_wp - current_pos
        dist = np.linalg.norm(diff)
        if dist < 1.0:
            return np.zeros(3), 0.0, "WAYPOINT_REACHED"

        unit_dir = diff / dist
        v_cmd = unit_dir * self.v_nominal
        # Desired heading points along geometric flight path
        desired_yaw = np.arctan2(unit_dir[1], unit_dir[0])
        return v_cmd, 0.0, "GEOMETRIC_CRUISE"
