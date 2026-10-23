"""
Model Predictive Path Integral (MPPI) Local Trajectory Planner (PRD Section 1.5 & Layer 5).
Executes K=1024 parallel sampled rollouts over a 6.0s horizon (N=30, dt=0.2s) at 20 Hz
optimizing electrical power, camera vibration stability, progress, and terrain clearance.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch


class MPPIPlanner:
    """
    GPU/CPU PyTorch-accelerated MPPI Trajectory Optimizer.
    """

    def __init__(
        self,
        num_rollouts_k: int = 1024,
        horizon_steps_n: int = 30,
        dt: float = 0.2,
        temperature_lambda: float = 0.5,
        w_progress: float = 15.0,
        w_energy: float = 0.04,
        w_camera_jitter: float = 8.0,
        w_wind_hazard: float = 12.0,
        w_obstacle_clearance: float = 60.0,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.K = num_rollouts_k
        self.N = horizon_steps_n
        self.dt = dt
        self.lambda_param = temperature_lambda

        # Cost weights
        self.w_progress = w_progress
        self.w_energy = w_energy
        self.w_camera = w_camera_jitter
        self.w_wind = w_wind_hazard
        self.w_obstacle = w_obstacle_clearance

        # Control limits: [vx, vy, vz, yaw_rate]
        self.v_xy_max = 14.0
        self.v_z_up_max = 4.0
        self.v_z_down_max = 3.0
        self.yaw_rate_max = 0.8

        # Noise standard deviations for control sampling
        self.control_std = torch.tensor([1.2, 1.2, 0.7, 0.25], device=self.device)

        # Baseline control sequence over horizon (N, 4)
        self.u_sequence = torch.zeros((self.N, 4), device=self.device)

    def optimize(
        self,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        target_setpoint: np.ndarray,
        power_model,
        camera_model,
        wind_field,
        gp_spatial_model,
        canyon_env,
        state_context,
    ) -> Tuple[np.ndarray, float, torch.Tensor]:
        """
        Executes one MPPI optimization iteration over K=1024 rollouts.

        Returns:
            optimal_v_cmd: [vx, vy, vz] for immediate execution
            optimal_yaw_rate: yaw rate command
            sampled_trajectories: (K, N, 3) trajectory tensor for 3D visualization
        """
        # 1. Sample control perturbations: (K, N, 4) ~ N(0, Sigma)
        noise = torch.randn((self.K, self.N, 4), device=self.device) * self.control_std

        # Perturbed control candidates
        u_candidates = self.u_sequence.unsqueeze(0) + noise

        # Clamp control candidates to flight envelope
        u_candidates[..., 0] = torch.clamp(u_candidates[..., 0], -self.v_xy_max, self.v_xy_max)
        u_candidates[..., 1] = torch.clamp(u_candidates[..., 1], -self.v_xy_max, self.v_xy_max)
        u_candidates[..., 2] = torch.clamp(u_candidates[..., 2], -self.v_z_down_max, self.v_z_up_max)

        # Stable capture mode angular rate restriction (PRD Section 4.4)
        yaw_limit = state_context.max_angular_rate_rad_s
        u_candidates[..., 3] = torch.clamp(u_candidates[..., 3], -yaw_limit, yaw_limit)

        # 2. Rollout forward kinematics over N steps
        pos_rollouts = torch.zeros((self.K, self.N + 1, 3), device=self.device)
        vel_rollouts = torch.zeros((self.K, self.N + 1, 3), device=self.device)

        pos_0 = torch.tensor(current_pos, dtype=torch.float32, device=self.device)
        vel_0 = torch.tensor(current_vel, dtype=torch.float32, device=self.device)

        pos_rollouts[:, 0, :] = pos_0
        vel_rollouts[:, 0, :] = vel_0

        # Dynamics step: simple high-fidelity multirotor tracking model
        kp_vel = 2.5
        for t in range(self.N):
            v_curr = vel_rollouts[:, t, :]
            v_cmd = u_candidates[:, t, :3]
            accel = kp_vel * (v_cmd - v_curr)
            v_next = v_curr + accel * self.dt
            p_next = pos_rollouts[:, t, :] + v_next * self.dt

            vel_rollouts[:, t + 1, :] = v_next
            pos_rollouts[:, t + 1, :] = p_next

        # 3. Cost Evaluation over all (K, N) steps
        target_t = torch.tensor(target_setpoint, dtype=torch.float32, device=self.device)

        # (a) Progress cost: distance from final predicted position to target setpoint
        final_pos = pos_rollouts[:, -1, :]
        dist_to_goal = torch.norm(final_pos - target_t, dim=-1)
        cost_progress = self.w_progress * dist_to_goal

        # (b) Wind & Relative Velocity for Power and Wind Hazard
        mid_positions = pos_rollouts[:, 1:, :]  # (K, N, 3)
        wind_batch = wind_field.get_wind_batch_torch(mid_positions)  # (K, N, 3)
        v_rel_batch = vel_rollouts[:, 1:, :] - wind_batch  # (K, N, 3)

        # Instantaneous power in Watts
        power_batch = power_model.compute_power_torch(v_rel_batch)  # (K, N)
        cost_power = self.w_energy * torch.sum(power_batch * (self.dt / 3600.0), dim=-1)

        # (c) Camera Stability Cost (PRD Section 3.4)
        tilt_est = torch.atan2(
            torch.norm(vel_rollouts[:, 1:, :2], dim=-1),
            torch.tensor(9.81, device=self.device),
        )
        omega_est = torch.stack([
            u_candidates[..., 3] * 0.2,
            u_candidates[..., 3] * 0.2,
            u_candidates[..., 3],
        ], dim=-1)
        wind_deriv_dummy = torch.zeros_like(omega_est)
        camera_costs = camera_model.evaluate_cost_batch_torch(
            omega_est,
            tilt_est,
            wind_deriv_dummy,
            is_stable_capture_mode=(state_context.current_mode == "STABLE_CAPTURE"),
        )
        cost_camera = self.w_camera * torch.sum(camera_costs * self.dt, dim=-1)

        # (d) Wind Hazard Penalty
        wind_speeds = torch.norm(wind_batch, dim=-1)
        wind_hazard_excess = torch.clamp(wind_speeds - 12.0, min=0.0)
        cost_wind = self.w_wind * torch.sum(wind_hazard_excess, dim=-1)

        # (e) Terrain & Obstacle Clearance
        # Sample canyon terrain height under predicted positions
        pos_np_x = pos_rollouts[:, 1:, 0].cpu().numpy()
        pos_np_y = pos_rollouts[:, 1:, 1].cpu().numpy()
        elevations = canyon_env.get_elevation(pos_np_x, pos_np_y)
        elev_t = torch.tensor(elevations, device=self.device, dtype=torch.float32)
        clearance = pos_rollouts[:, 1:, 2] - elev_t
        terrain_violation = torch.clamp(12.0 - clearance, min=0.0)
        cost_obstacle = self.w_obstacle * torch.sum(terrain_violation ** 2, dim=-1)

        # Total cost per rollout
        total_costs = cost_progress + cost_power + cost_camera + cost_wind + cost_obstacle

        # 4. Model Predictive Path Integral Weighting
        min_cost = torch.min(total_costs)
        weights = torch.exp(- (total_costs - min_cost) / self.lambda_param)
        weights = weights / (torch.sum(weights) + 1e-8)  # (K,)

        # Compute updated control sequence
        # (N, 4) = sum_k weights[k] * u_candidates[k]
        u_optimal_seq = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * u_candidates, dim=0)

        # Shift sequence for receding horizon warm start
        self.u_sequence[:-1] = u_optimal_seq[1:].clone()
        self.u_sequence[-1] = u_optimal_seq[-1].clone()

        # Immediate control command to execute
        u_exec = u_optimal_seq[0].cpu().numpy()
        v_cmd = u_exec[:3]
        yaw_rate_cmd = float(u_exec[3])

        return v_cmd, yaw_rate_cmd, pos_rollouts
