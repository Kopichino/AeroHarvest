"""
6-DOF Multirotor Hexacopter Dynamics & IMU Telemetry Generator.
Integrates translational equations of motion, aerodynamic parasite drag,
attitude kinematics, and synthetic 50 Hz noisy telemetry output for PX4/ROS 2.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np


@dataclass
class HexacopterState:
    position: np.ndarray  # [x, y, z] in meters (Inertial ENU/NED)
    velocity: np.ndarray  # [vx, vy, vz] in m/s
    attitude_rot: np.ndarray  # 3x3 rotation matrix R_b^I (body to inertial)
    angular_velocity: np.ndarray  # [p, q, r] in rad/s (body frame)
    tilt_angle_rad: float  # total lean angle theta_tilt = arccos(R_33)
    yaw_rad: float
    thrust_n: float  # current total motor thrust in Newtons
    battery_energy_wh: float  # current stored battery energy in Wh
    battery_capacity_wh: float = 120.0


class HexacopterDynamics:
    """
    Simulation of 2.4 kg hexacopter multirotor subject to external wind forces.
    """

    def __init__(
        self,
        mass_kg: float = 2.4,
        drag_coeff_cd: float = 0.45,
        projected_area_m2: float = 0.12,
        air_density: float = 1.225,
        battery_wh: float = 120.0,
        imu_accel_noise_std: float = 0.08,
        telemetry_thrust_noise_std: float = 0.25,
    ):
        self.mass = mass_kg
        self.g = 9.80665
        self.rho = air_density
        self.cd = drag_coeff_cd
        self.area = projected_area_m2
        self.battery_capacity_wh = battery_wh
        self.accel_noise_std = imu_accel_noise_std
        self.thrust_noise_std = telemetry_thrust_noise_std

        # Drag constant: 0.5 * rho * Cd * A
        self.drag_factor = 0.5 * self.rho * self.cd * self.area

        # State initialization
        self.reset(np.array([0.0, -450.0, 35.0]))

    def reset(self, initial_position: np.ndarray):
        self.state = HexacopterState(
            position=initial_position.astype(np.float64).copy(),
            velocity=np.zeros(3, dtype=np.float64),
            attitude_rot=np.eye(3, dtype=np.float64),
            angular_velocity=np.zeros(3, dtype=np.float64),
            tilt_angle_rad=0.0,
            yaw_rad=0.0,
            thrust_n=self.mass * self.g,
            battery_energy_wh=self.battery_capacity_wh,
            battery_capacity_wh=self.battery_capacity_wh,
        )

    def step(
        self,
        command_velocity: np.ndarray,
        command_yaw_rate: float,
        wind_vector: np.ndarray,
        dt: float,
        power_consumption_w: float = 405.0,
    ) -> Dict[str, Any]:
        """
        Executes one physics integration step.
        command_velocity: Desired ground velocity [vx, vy, vz]
        command_yaw_rate: Desired yaw rate d(psi)/dt in rad/s
        wind_vector: True local wind [vwx, vwy, vwz] in m/s
        dt: Timestep in seconds
        """
        pos = self.state.position
        vel = self.state.velocity

        # Relative air velocity: v_rel = v_ground - v_wind
        v_rel = vel - wind_vector
        v_rel_norm = np.linalg.norm(v_rel)

        # Aerodynamic parasite drag force: F_drag = -0.5 * rho * Cd * A * ||v_rel|| * v_rel
        f_drag = -self.drag_factor * v_rel_norm * v_rel

        # Flight controller closed-loop tracking (PX4 velocity tracking inner-loop model)
        # Desired inertial acceleration: a_des = K_p * (v_cmd - v)
        kp_vel = 2.5
        a_des = kp_vel * (command_velocity - vel)
        # Acceleration limits
        a_des[:2] = np.clip(a_des[:2], -6.0, 6.0)
        a_des[2] = np.clip(a_des[2], -3.0, 4.0)

        # Total force required: F_total = m * a_des + m * g * e3 - F_drag
        f_req = self.mass * a_des + np.array([0.0, 0.0, self.mass * self.g]) - f_drag

        # Thrust magnitude and orientation
        thrust_mag = np.linalg.norm(f_req)
        thrust_mag = np.clip(thrust_mag, 0.2 * self.mass * self.g, 2.5 * self.mass * self.g)
        self.state.thrust_n = thrust_mag

        # Thrust unit direction in inertial frame z_b_I
        z_body = f_req / (thrust_mag + 1e-6)

        # Yaw control
        new_yaw = self.state.yaw_rad + command_yaw_rate * dt
        self.state.yaw_rad = np.arctan2(np.sin(new_yaw), np.cos(new_yaw))

        # Reconstruct body orientation matrix R_b^I
        # Heading reference vector in horizontal plane
        x_c = np.array([np.cos(self.state.yaw_rad), np.sin(self.state.yaw_rad), 0.0])
        y_body = np.cross(z_body, x_c)
        y_norm = np.linalg.norm(y_body)
        if y_norm < 1e-4:
            y_body = np.array([0.0, 1.0, 0.0])
        else:
            y_body /= y_norm
        x_body = np.cross(y_body, z_body)

        R_b_I = np.column_stack([x_body, y_body, z_body])
        self.state.attitude_rot = R_b_I

        # Tilt lean angle: theta = arccos(R_33)
        tilt = np.arccos(np.clip(R_b_I[2, 2], -1.0, 1.0))
        self.state.tilt_angle_rad = tilt

        # Angular velocity approximation: difference between consecutive rotation matrices
        # Body frame angular rate [p, q, r]
        omega_body = np.array([
            (command_velocity[1] - vel[1]) * 0.15,
            (command_velocity[0] - vel[0]) * 0.15,
            command_yaw_rate,
        ])
        self.state.angular_velocity = omega_body

        # Actual acceleration: a = (R_b^I * [0, 0, T]^T + F_drag)/m - g * e3
        f_thrust_inertial = R_b_I @ np.array([0.0, 0.0, thrust_mag])
        actual_accel = (f_thrust_inertial + f_drag) / self.mass - np.array([0.0, 0.0, self.g])

        # Integrate velocity and position
        self.state.velocity += actual_accel * dt
        self.state.position += self.state.velocity * dt

        # Update battery energy consumption
        # Wh = W * (dt / 3600.0)
        energy_used_wh = power_consumption_w * (dt / 3600.0)
        self.state.battery_energy_wh = max(0.0, self.state.battery_energy_wh - energy_used_wh)

        # 50 Hz synthetic noisy telemetry generation for disturbance observer:
        # a_meas = actual_accel + g * e3 (specific force sensed by accelerometer) + noise
        specific_force = actual_accel + np.array([0.0, 0.0, self.g])
        noise_a = np.random.normal(0.0, self.accel_noise_std, size=3)
        a_meas = specific_force + noise_a

        noise_t = np.random.normal(0.0, self.thrust_noise_std)
        t_telemetry = max(0.0, thrust_mag + noise_t)

        telemetry = {
            "a_meas": a_meas,
            "R_b_I": R_b_I.copy(),
            "thrust_telemetry": t_telemetry,
            "v_ground": self.state.velocity.copy(),
            "position": self.state.position.copy(),
            "tilt_angle_deg": np.degrees(tilt),
            "angular_velocity": omega_body.copy(),
            "battery_energy_wh": self.state.battery_energy_wh,
            "battery_percentage": (self.state.battery_energy_wh / self.battery_capacity_wh) * 100.0,
        }

        return telemetry
