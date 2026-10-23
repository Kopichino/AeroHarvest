"""
50 Hz Telemetry-Based Disturbance Observer for Multirotor UAVs.
Inverts the multirotor body parasite drag equation from IMU specific force/acceleration,
attitude rotation matrix, and motor thrust telemetry to estimate 3D wind vectors in real time
without requiring dedicated pitot hardware.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


class TelemetryDisturbanceObserver:
    """
    Mathematical Implementation of PRD Section 3.1.
    Estimates inertial wind vector v_hat_wind = v_ground - v_rel.
    """

    def __init__(
        self,
        mass_kg: float = 2.4,
        air_density: float = 1.225,
        drag_coeff_cd: float = 0.45,
        projected_area_m2: float = 0.12,
        filter_alpha: float = 0.25,  # exponential smoothing factor at 50 Hz
    ):
        self.mass = mass_kg
        self.rho = air_density
        self.cd = drag_coeff_cd
        self.area = projected_area_m2
        self.g = 9.80665
        self.alpha = filter_alpha

        # Precompute denominator: rho * Cd * A
        self.drag_factor = self.rho * self.cd * self.area

        # State memory
        self.v_wind_filtered: np.ndarray = np.zeros(3, dtype=np.float64)
        self.initialized: bool = False

    def update(
        self,
        a_meas: np.ndarray,
        R_b_I: np.ndarray,
        thrust_telemetry: float,
        v_ground: np.ndarray,
        is_specific_force: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Executes one disturbance observer calculation step at 50 Hz.

        Args:
            a_meas: Accelerometer measurement [ax, ay, az] (m/s^2)
            R_b_I: Body-to-Inertial rotation matrix (3x3)
            thrust_telemetry: Total thrust setpoint / telemetry (Newtons)
            v_ground: Vehicle ground velocity vector [vx, vy, vz] (m/s)
            is_specific_force: True if accelerometer measures specific force (a + g*e3)

        Returns:
            v_wind_est: Filtered estimated 3D wind velocity vector in inertial frame [vx, vy, vz]
            v_rel_est: Estimated relative airspeed vector [u, v, w]
            F_ext: Estimated external aerodynamic drag force vector [Fx, Fy, Fz]
        """
        thrust_body = np.array([0.0, 0.0, float(thrust_telemetry)], dtype=np.float64)
        thrust_inertial = R_b_I @ thrust_body

        if is_specific_force:
            # Specific force a_spec = a_inertial + g * e3
            # F_ext = m * a_spec - thrust_inertial
            f_ext = self.mass * a_meas - thrust_inertial
        else:
            # PRD Equation 104: F_ext = m * a_meas - R_b^I * [0, 0, T]^T + m * g * e3
            f_ext = self.mass * a_meas - thrust_inertial + np.array([0.0, 0.0, self.mass * self.g])

        f_ext_norm = np.linalg.norm(f_ext)

        # Invert body drag equation (PRD Eq 114)
        # v_rel = - sqrt(2 * ||F_ext|| / (rho * Cd * A)) * (F_ext / ||F_ext||)
        if f_ext_norm < 1e-4:
            v_rel = np.zeros(3, dtype=np.float64)
        else:
            speed_rel = np.sqrt((2.0 * f_ext_norm) / (self.drag_factor + 1e-6))
            # Speed clamping to physical multirotor envelope
            speed_rel = min(speed_rel, 30.0)
            v_rel = -speed_rel * (f_ext / f_ext_norm)

        # Invert for wind velocity (PRD Eq 120): v_hat_wind = v_ground - v_rel
        v_wind_raw = v_ground - v_rel

        # Exponential moving average filter to reject motor vibration spikes
        if not self.initialized:
            self.v_wind_filtered = v_wind_raw.copy()
            self.initialized = True
        else:
            self.v_wind_filtered = self.alpha * v_wind_raw + (1.0 - self.alpha) * self.v_wind_filtered

        return self.v_wind_filtered.copy(), v_rel, f_ext

    def get_estimate(self) -> np.ndarray:
        return self.v_wind_filtered.copy()
