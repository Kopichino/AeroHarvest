"""
Baseline 3: GP Energy-Only Planner (PRD Section 6.2).
Uses the Gaussian Process belief map and multirotor electrical power model
to optimize energy paths, but omits Active Informative Probing, Camera Stability,
and Dynamic Confidence-Gated Battery Reserves.
"""

from typing import Tuple, Dict, Any
import numpy as np
import torch
from aeroharvest.planning.mppi_planner import MPPIPlanner
from aeroharvest.planning.state_machine import StateMachineContext, OperationalMode


class GPEnergyOnlyBaseline:
    """
    GP-Energy baseline: optimizes trajectory for energy/power using the GP spatial map,
    but without active probing detours, camera stabilization, or confidence battery gates.
    """

    def __init__(self, mppi_planner: MPPIPlanner, static_rtb_percentage: float = 18.0):
        self.mppi = mppi_planner
        self.rtb_thresh = static_rtb_percentage

    def step(
        self,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        target_wp: np.ndarray,
        launch_point: np.ndarray,
        battery_percentage: float,
        power_model,
        camera_model,
        wind_field,
        gp_spatial_model,
        canyon_env,
    ) -> Tuple[np.ndarray, float, str]:
        # Static RTB threshold without confidence inflation
        if battery_percentage <= self.rtb_thresh:
            target_pt = launch_point
            mode_str = "RETURN_TO_BASE_STATIC"
        else:
            target_pt = target_wp
            mode_str = "GP_ENERGY_CRUISE"

        # Context without camera stabilization restrictions
        dummy_context = StateMachineContext(
            current_mode=OperationalMode.MISSION_CRUISE,
            target_setpoint=target_pt,
            speed_target_m_s=8.0,
            max_tilt_deg=25.0,
            max_angular_rate_rad_s=0.8,
            mode_reason="Energy-only trajectory optimization",
        )

        # Run MPPI with zero camera jitter weight (only energy + progress)
        saved_w_camera = self.mppi.w_camera
        self.mppi.w_camera = 0.0

        v_cmd, yaw_rate_cmd, _ = self.mppi.optimize(
            current_pos=current_pos,
            current_vel=current_vel,
            target_setpoint=target_pt,
            power_model=power_model,
            camera_model=camera_model,
            wind_field=wind_field,
            gp_spatial_model=gp_spatial_model,
            canyon_env=canyon_env,
            state_context=dummy_context,
        )

        self.mppi.w_camera = saved_w_camera
        return v_cmd, yaw_rate_cmd, mode_str
