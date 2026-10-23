"""
Tactical Decision Engine & Operational State Machine (PRD Section 4).
Manages transitions across 5 operational flight modes:
1. MISSION_CRUISE
2. WIND_ADAPT
3. ACTIVE_PROBE
4. STABLE_CAPTURE
5. RETURN_TO_BASE
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import numpy as np


class OperationalMode(str, Enum):
    INITIALIZATION = "INITIALIZATION"
    MISSION_CRUISE = "MISSION_CRUISE"
    WIND_ADAPT = "WIND_ADAPT"
    ACTIVE_PROBE = "ACTIVE_PROBE"
    STABLE_CAPTURE = "STABLE_CAPTURE"
    RETURN_TO_BASE = "RETURN_TO_BASE"


@dataclass
class StateMachineContext:
    current_mode: OperationalMode
    target_setpoint: np.ndarray
    speed_target_m_s: float
    max_tilt_deg: float
    max_angular_rate_rad_s: float
    mode_reason: str
    active_probe_duration_left: float = 0.0


class TacticalStateMachine:
    """
    State machine executing PRD Section 4 mode switching logic.
    """

    def __init__(
        self,
        wind_speed_adapt_threshold: float = 4.0,
        uncertainty_probe_threshold: float = 0.8,
        nominal_uncertainty_threshold: float = 0.2,
        info_efficiency_threshold: float = 1.2,
    ):
        self.wind_adapt_thresh = wind_speed_adapt_threshold
        self.probe_uncertainty_thresh = uncertainty_probe_threshold
        self.nominal_uncertainty_thresh = nominal_uncertainty_threshold
        self.info_eff_thresh = info_efficiency_threshold

        self.current_mode = OperationalMode.INITIALIZATION
        self.active_probe_timer = 0.0
        self.probe_waypoint: Optional[np.ndarray] = None

    def update(
        self,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        wind_est: np.ndarray,
        local_uncertainty_var: float,
        battery_wh: float,
        battery_reserve_wh: float,
        active_thermal_target,
        primary_waypoint: np.ndarray,
        launch_point: np.ndarray,
        dt: float,
    ) -> StateMachineContext:
        """
        Evaluates conditions and returns updated mode & control constraints.
        """
        wind_speed = float(np.linalg.norm(wind_est))

        # 1. PRIORITY FAILSAFE: Return to Base Gate (PRD Section 4.5 & 3.6)
        if battery_wh <= battery_reserve_wh:
            self.current_mode = OperationalMode.RETURN_TO_BASE
            return StateMachineContext(
                current_mode=OperationalMode.RETURN_TO_BASE,
                target_setpoint=launch_point.copy(),
                speed_target_m_s=8.5,
                max_tilt_deg=22.0,
                max_angular_rate_rad_s=0.6,
                mode_reason=f"Confidence-Gated Battery Reserve reached ({battery_wh:.1f} Wh <= {battery_reserve_wh:.1f} Wh)",
            )

        # 2. PRIORITY INSPECTION: Stable Capture Mode (PRD Section 4.4)
        if active_thermal_target is not None and not active_thermal_target.completed:
            self.current_mode = OperationalMode.STABLE_CAPTURE
            return StateMachineContext(
                current_mode=OperationalMode.STABLE_CAPTURE,
                target_setpoint=active_thermal_target.position.copy(),
                speed_target_m_s=2.5,
                max_tilt_deg=12.0,  # PRD Section 4.4: tilt <= 12 deg
                max_angular_rate_rad_s=0.1,  # PRD Section 4.4: omega <= 0.1 rad/s
                mode_reason=f"Thermal inspection target {active_thermal_target.target_id} capture active",
            )

        # 3. Handle ongoing active probe detour
        if self.current_mode == OperationalMode.ACTIVE_PROBE:
            self.active_probe_timer -= dt
            if self.active_probe_timer > 0.0 and self.probe_waypoint is not None:
                return StateMachineContext(
                    current_mode=OperationalMode.ACTIVE_PROBE,
                    target_setpoint=self.probe_waypoint.copy(),
                    speed_target_m_s=6.0,
                    max_tilt_deg=20.0,
                    max_angular_rate_rad_s=0.5,
                    mode_reason=f"Executing exploratory wind-mapping detour (time left: {self.active_probe_timer:.1f}s)",
                    active_probe_duration_left=self.active_probe_timer,
                )
            else:
                self.current_mode = OperationalMode.MISSION_CRUISE

        # 4. ACTIVE_PROBE Trigger Check (PRD Section 4.3)
        # Condition: local GP variance > 0.8 m^2/s^2 AND info efficiency > threshold
        if local_uncertainty_var > self.probe_uncertainty_thresh:
            # Generate 35-50m lateral exploratory detour perpendicular to mission leg
            leg_dir = primary_waypoint - current_pos
            leg_dist = np.linalg.norm(leg_dir[:2])
            if leg_dist > 40.0:
                unit_leg = leg_dir / (np.linalg.norm(leg_dir) + 1e-6)
                # Lateral perpendicular offset
                lateral = np.array([-unit_leg[1], unit_leg[0], 0.25])
                probe_pt = current_pos + 45.0 * lateral
                # Check terrain safety
                if probe_pt[2] > 40.0:
                    self.current_mode = OperationalMode.ACTIVE_PROBE
                    self.probe_waypoint = probe_pt
                    self.active_probe_timer = 7.0
                    return StateMachineContext(
                        current_mode=OperationalMode.ACTIVE_PROBE,
                        target_setpoint=probe_pt.copy(),
                        speed_target_m_s=6.0,
                        max_tilt_deg=20.0,
                        max_angular_rate_rad_s=0.5,
                        mode_reason=f"High airspace uncertainty (sigma^2={local_uncertainty_var:.2f} > {self.probe_uncertainty_thresh})",
                        active_probe_duration_left=self.active_probe_timer,
                    )

        # 5. WIND_ADAPT Mode (PRD Section 4.2)
        # Condition: wind speed >= 4.0 m/s
        if wind_speed >= self.wind_adapt_thresh:
            self.current_mode = OperationalMode.WIND_ADAPT
            # In WIND_ADAPT, target trajectory shifts into tailwind corridor / canyon draft
            # Compute draft-assisted waypoint
            draft_bias = 0.3 * wind_est
            adapt_waypoint = primary_waypoint + draft_bias
            return StateMachineContext(
                current_mode=OperationalMode.WIND_ADAPT,
                target_setpoint=adapt_waypoint,
                speed_target_m_s=9.0,
                max_tilt_deg=25.0,
                max_angular_rate_rad_s=0.6,
                mode_reason=f"Exploiting wind draft (wind={wind_speed:.1f} m/s >= {self.wind_adapt_thresh} m/s)",
            )

        # 6. MISSION_CRUISE Mode (PRD Section 4.1)
        self.current_mode = OperationalMode.MISSION_CRUISE
        return StateMachineContext(
            current_mode=OperationalMode.MISSION_CRUISE,
            target_setpoint=primary_waypoint.copy(),
            speed_target_m_s=8.0,
            max_tilt_deg=20.0,
            max_angular_rate_rad_s=0.5,
            mode_reason=f"Nominal cruise navigation (wind={wind_speed:.1f} m/s, var={local_uncertainty_var:.2f})",
        )
