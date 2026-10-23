"""
AeroHarvest Full Autonomy Stack.
Closed-loop integration of all five runtime layers:
Layer 1: Simulation Physics (Hexacopter Dynamics & Wind Field)
Layer 2: 50 Hz Telemetry Disturbance Observer
Layer 3: GPyTorch Sparse Gaussian Process (M=128 Inducing Points, Matern 5/2)
Layer 4: Tactical State Machine & Confidence-Gated Battery Reserve
Layer 5: CUDA/CPU MPPI Trajectory Optimizer
"""

from typing import Dict, Any, List, Tuple, Optional
import time
import numpy as np

from aeroharvest.physics.canyon_environment import MountainCanyon, MissionScenario
from aeroharvest.physics.wind_field import DynamicWindField
from aeroharvest.physics.hexacopter_dynamics import HexacopterDynamics
from aeroharvest.perception.disturbance_observer import TelemetryDisturbanceObserver
from aeroharvest.perception.sparse_gp import SpatialWindGaussianProcess
from aeroharvest.models.power_model import MultirotorPowerModel
from aeroharvest.models.camera_jitter_model import CameraStabilityModel
from aeroharvest.models.mutual_information import MutualInformationMetric
from aeroharvest.planning.battery_reserve import ConfidenceGatedBatteryReserve
from aeroharvest.planning.decision_scoring import TacticalDecisionScorer
from aeroharvest.planning.state_machine import TacticalStateMachine, OperationalMode, StateMachineContext
from aeroharvest.planning.mppi_planner import MPPIPlanner


class AeroHarvestStack:
    """
    Main closed-loop autonomy system integrating all 5 layers.
    """

    def __init__(
        self,
        base_wind_speed: float = 6.0,
        base_wind_heading_deg: float = 45.0,
        gust_amplitude: float = 3.0,
        seed: int = 42,
        use_gp: bool = True,
        use_mppi: bool = True,
        use_confidence_reserve: bool = True,
        use_camera_stabilization: bool = True,
    ):
        # Configuration flags for ablation/baseline support
        self.use_gp = use_gp
        self.use_mppi = use_mppi
        self.use_confidence_reserve = use_confidence_reserve
        self.use_camera_stabilization = use_camera_stabilization

        # Physics & Environment
        self.canyon = MountainCanyon()
        self.scenario = MissionScenario(self.canyon)
        self.wind_field = DynamicWindField(
            base_wind_speed=base_wind_speed,
            base_wind_heading_deg=base_wind_heading_deg,
            gust_amplitude=gust_amplitude,
            seed=seed,
        )
        self.vehicle = HexacopterDynamics()
        self.vehicle.reset(self.scenario.launch_point)

        # Perception
        self.observer = TelemetryDisturbanceObserver()
        self.gp_model = SpatialWindGaussianProcess(num_inducing=128, lengthscale=40.0)

        # Mathematical Models
        self.power_model = MultirotorPowerModel()
        self.camera_model = CameraStabilityModel()
        self.info_metric = MutualInformationMetric()

        # Decision & Planning
        self.battery_gate = ConfidenceGatedBatteryReserve()
        self.decision_scorer = TacticalDecisionScorer()
        self.state_machine = TacticalStateMachine()
        self.mppi = MPPIPlanner(num_rollouts_k=1024, horizon_steps_n=30, dt=0.2)

        # Timing and state
        self.sim_time = 0.0
        self.dt_physics = 0.02  # 50 Hz inner physics & observer loop
        self.mppi_period = 0.05  # 20 Hz control loop
        self.mppi_timer = 0.0
        self.gp_fit_timer = 0.0

        # Cached control commands
        self.current_v_cmd = np.zeros(3)
        self.current_yaw_rate_cmd = 0.0
        self.current_rollouts = None
        self.last_loop_latency_ms = 0.0

        # Telemetry and logging history
        self.history: Dict[str, List] = {
            "time": [],
            "position": [],
            "velocity": [],
            "wind_true": [],
            "wind_est": [],
            "wind_uncertainty": [],
            "battery_wh": [],
            "battery_pct": [],
            "battery_reserve_wh": [],
            "power_w": [],
            "tilt_deg": [],
            "angular_velocity": [],
            "mode": [],
            "latency_ms": [],
        }

    def step(self) -> Dict[str, Any]:
        """
        Executes one 50 Hz step of the closed-loop system.
        """
        t_start = time.perf_counter()
        pos = self.vehicle.state.position
        vel = self.vehicle.state.velocity

        # 1. Update Mission scenario (targets and waypoints)
        self.scenario.update_targets(pos, self.dt_physics)
        active_target = self.scenario.get_active_thermal_target(pos)
        self.scenario.check_waypoint_reached(pos)
        current_wp = self.scenario.get_current_waypoint()

        # 2. Get true wind from environment
        true_wind = self.wind_field.get_wind_at(pos, self.sim_time)
        wind_deriv = self.wind_field.get_wind_derivative(pos, self.sim_time)

        # 3. Physics step: hexacopter moves subject to true wind and current control command
        # Calculate instant power for battery drawdown
        v_rel = vel - true_wind
        instant_power = self.power_model.compute_power_np(v_rel)
        telemetry = self.vehicle.step(
            command_velocity=self.current_v_cmd,
            command_yaw_rate=self.current_yaw_rate_cmd,
            wind_vector=true_wind,
            dt=self.dt_physics,
            power_consumption_w=instant_power,
        )

        # 4. Layer 2: 50 Hz Disturbance Observer
        wind_est, v_rel_est, f_ext = self.observer.update(
            a_meas=telemetry["a_meas"],
            R_b_I=telemetry["R_b_I"],
            thrust_telemetry=telemetry["thrust_telemetry"],
            v_ground=telemetry["v_ground"],
        )

        # Feed observation into GP belief map
        if self.use_gp:
            self.gp_model.add_observation(pos, wind_est)
            # Periodic variational fit
            self.gp_fit_timer += self.dt_physics
            if self.gp_fit_timer >= 0.5:
                self.gp_model.fit_online(steps=1)
                self.gp_fit_timer = 0.0

        # Query local spatial uncertainty
        if self.use_gp:
            _, local_var_arr = self.gp_model.predict(pos)
            local_uncertainty_var = float(local_var_arr[0])
        else:
            local_uncertainty_var = 0.0

        # 5. Layer 4: Battery Reserve Gate & State Machine (at 20 Hz or on timer)
        if self.use_confidence_reserve and self.use_gp:
            reserve_wh, _, _ = self.battery_gate.evaluate_reserve_threshold(
                current_position=pos,
                launch_position=self.scenario.launch_point,
                gp_spatial_model=self.gp_model,
            )
        else:
            # Fixed 15% static reserve
            reserve_wh = 0.15 * self.vehicle.battery_capacity_wh

        state_context = self.state_machine.update(
            current_pos=pos,
            current_vel=vel,
            wind_est=wind_est,
            local_uncertainty_var=local_uncertainty_var,
            battery_wh=self.vehicle.state.battery_energy_wh,
            battery_reserve_wh=reserve_wh,
            active_thermal_target=active_target,
            primary_waypoint=current_wp,
            launch_point=self.scenario.launch_point,
            dt=self.dt_physics,
        )

        # 6. Layer 5: 20 Hz MPPI Local Trajectory Optimization
        self.mppi_timer += self.dt_physics
        if self.mppi_timer >= self.mppi_period:
            self.mppi_timer = 0.0

            if self.use_mppi:
                v_cmd, yaw_cmd, rollouts = self.mppi.optimize(
                    current_pos=pos,
                    current_vel=vel,
                    target_setpoint=state_context.target_setpoint,
                    power_model=self.power_model,
                    camera_model=self.camera_model,
                    wind_field=self.wind_field,
                    gp_spatial_model=self.gp_model,
                    canyon_env=self.canyon,
                    state_context=state_context,
                )
                self.current_v_cmd = v_cmd
                self.current_yaw_rate_cmd = yaw_cmd
                self.current_rollouts = rollouts
            else:
                # Geometric setpoint tracking (MPPI disabled ablation)
                diff = state_context.target_setpoint - pos
                d = np.linalg.norm(diff)
                speed = min(state_context.speed_target_m_s, d * 1.5)
                self.current_v_cmd = (diff / max(1e-3, d)) * speed
                self.current_yaw_rate_cmd = 0.0

        t_end = time.perf_counter()
        self.last_loop_latency_ms = (t_end - t_start) * 1000.0

        # Log history
        self.sim_time += self.dt_physics
        self.history["time"].append(self.sim_time)
        self.history["position"].append(pos.copy())
        self.history["velocity"].append(vel.copy())
        self.history["wind_true"].append(true_wind.copy())
        self.history["wind_est"].append(wind_est.copy())
        self.history["wind_uncertainty"].append(local_uncertainty_var)
        self.history["battery_wh"].append(self.vehicle.state.battery_energy_wh)
        self.history["battery_pct"].append(telemetry["battery_percentage"])
        self.history["battery_reserve_wh"].append(reserve_wh)
        self.history["power_w"].append(instant_power)
        self.history["tilt_deg"].append(telemetry["tilt_angle_deg"])
        self.history["angular_velocity"].append(telemetry["angular_velocity"].copy())
        self.history["mode"].append(state_context.current_mode.value)
        self.history["latency_ms"].append(self.last_loop_latency_ms)

        step_info = {
            "time": self.sim_time,
            "position": pos,
            "velocity": vel,
            "wind_true": true_wind,
            "wind_est": wind_est,
            "uncertainty_var": local_uncertainty_var,
            "battery_pct": telemetry["battery_percentage"],
            "battery_wh": self.vehicle.state.battery_energy_wh,
            "battery_reserve_wh": reserve_wh,
            "power_w": instant_power,
            "tilt_deg": telemetry["tilt_angle_deg"],
            "mode": state_context.current_mode.value,
            "mode_reason": state_context.mode_reason,
            "is_mission_complete": self.scenario.is_mission_complete(),
            "active_target_id": active_target.target_id if active_target else None,
            "latency_ms": self.last_loop_latency_ms,
        }

        return step_info
