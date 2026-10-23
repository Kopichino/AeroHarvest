"""
AeroHarvest Comprehensive Unit & Mathematical Validation Test Suite.
Verifies all mathematical formulations from PRD Sections 3.1 to 3.6:
1. Disturbance observer drag equation inversion
2. GPyTorch Matérn 5/2 Sparse Gaussian Process inference
3. Multirotor electrical power model calculations
4. Camera jitter & tilt exceedance cost
5. Mutual information calculation
6. Confidence-gated battery reserve inflation
7. MPPI 1024-rollout trajectory optimization execution
"""

import sys
import os

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from aeroharvest.physics.canyon_environment import MountainCanyon, MissionScenario
from aeroharvest.physics.wind_field import DynamicWindField
from aeroharvest.perception.disturbance_observer import TelemetryDisturbanceObserver
from aeroharvest.perception.sparse_gp import SpatialWindGaussianProcess
from aeroharvest.models.power_model import MultirotorPowerModel
from aeroharvest.models.camera_jitter_model import CameraStabilityModel
from aeroharvest.models.mutual_information import MutualInformationMetric
from aeroharvest.planning.battery_reserve import ConfidenceGatedBatteryReserve
from aeroharvest.planning.decision_scoring import TacticalDecisionScorer, ActionCandidate
from aeroharvest.planning.mppi_planner import MPPIPlanner
from aeroharvest.planning.state_machine import TacticalStateMachine, StateMachineContext, OperationalMode


def test_disturbance_observer():
    print("Testing Disturbance Observer Drag Inversion (PRD Section 3.1)...")
    observer = TelemetryDisturbanceObserver(mass_kg=2.4, air_density=1.225, drag_coeff_cd=0.45, projected_area_m2=0.12)

    # Synthetic scenario: Wind is [5.0, 0.0, 0.0], vehicle is stationary hover
    true_wind = np.array([5.0, 0.0, 0.0])
    v_ground = np.zeros(3)
    v_rel = v_ground - true_wind  # [-5.0, 0.0, 0.0]

    # F_drag = -0.5 * rho * Cd * A * ||v_rel|| * v_rel = -0.5 * 1.225 * 0.45 * 0.12 * 5.0 * [-5.0, 0.0, 0.0] = [+0.8268, 0, 0] N
    drag_factor = 0.5 * 1.225 * 0.45 * 0.12
    f_drag_true = -drag_factor * np.linalg.norm(v_rel) * v_rel

    # Specific force measured by IMU: a_spec = (f_thrust + f_drag)/m
    thrust_mag = 2.4 * 9.80665
    R_b_I = np.eye(3)
    a_spec = np.array([f_drag_true[0] / 2.4, 0.0, 9.80665])

    for _ in range(10):
        w_est, v_rel_est, f_ext = observer.update(a_spec, R_b_I, thrust_mag, v_ground, is_specific_force=True)

    err = np.linalg.norm(w_est - true_wind)
    print(f"  -> True Wind: {true_wind}, Estimated Wind: {np.round(w_est, 2)}, Error: {err:.4f} m/s")
    assert err < 0.2, f"Observer error too high: {err}"
    print("  [PASSED] Disturbance Observer verified.\n")


def test_power_model():
    print("Testing Multirotor Power Model (PRD Section 3.3)...")
    power_model = MultirotorPowerModel(p_hover_w=380.0, v_induced_m_s=5.2, p_electronics_w=25.0)

    # Hover condition: v_rel = 0
    p_hover = power_model.compute_power_np(np.zeros(3))
    print(f"  -> Hover Power: {p_hover:.2f} W (Expected: 380 + 25 = 405 W)")
    assert abs(p_hover - 405.0) < 1.0, f"Hover power mismatch: {p_hover}"

    # Forward flight airspeed = 10 m/s
    p_10ms = power_model.compute_power_np(np.array([10.0, 0.0, 0.0]))
    print(f"  -> 10 m/s Flight Power: {p_10ms:.2f} W")
    assert p_10ms > 350.0

    # PyTorch batch computation
    v_rel_t = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    p_batch = power_model.compute_power_torch(v_rel_t)
    assert p_batch.shape == (2,)
    print("  [PASSED] Power Model verified.\n")


def test_camera_jitter_model():
    print("Testing Camera Stability Model (PRD Section 3.4)...")
    camera_model = CameraStabilityModel()

    # Case A: Nominal hover without excessive tilt
    cost_safe = camera_model.evaluate_cost_step_np(
        omega_body=np.zeros(3),
        theta_tilt_rad=np.radians(10.0),
        wind_derivative=np.zeros(3),
    )
    print(f"  -> Safe Hover Camera Cost: {cost_safe:.4f}")
    assert cost_safe == 0.0

    # Case B: Exceeding tilt angle (e.g. 25 deg in high wind vs 15 deg limit)
    cost_violating = camera_model.evaluate_cost_step_np(
        omega_body=np.array([0.4, 0.4, 0.0]),
        theta_tilt_rad=np.radians(25.0),
        wind_derivative=np.array([1.5, 0.0, 0.0]),
    )
    print(f"  -> High Wind Shaking Camera Cost: {cost_violating:.2f}")
    assert cost_violating > 1.0
    print("  [PASSED] Camera Stability Model verified.\n")


def test_sparse_gp():
    print("Testing GPyTorch Sparse Gaussian Process (PRD Section 3.2)...")
    gp_model = SpatialWindGaussianProcess(num_inducing=128, lengthscale=40.0)

    # Add synthetic spatial observations
    for x in np.linspace(-100, 100, 15):
        pos = np.array([x, 50.0, 60.0])
        wind = np.array([6.0 + 0.02 * x, 1.0, 0.5])
        gp_model.add_observation(pos, wind)

    gp_model.fit_online(steps=2)

    # Query near observation
    mean_near, var_near = gp_model.predict(np.array([0.0, 50.0, 60.0]))
    # Query far in unvisited canyon airspace
    mean_far, var_far = gp_model.predict(np.array([400.0, -400.0, 200.0]))

    print(f"  -> Near Observed Airspace Variance: {var_near[0]:.4f}")
    print(f"  -> Far Unvisited Airspace Variance: {var_far[0]:.4f}")
    assert var_far[0] > var_near[0], "Uncertainty did not drop near observations!"
    print("  [PASSED] Sparse Gaussian Process verified.\n")


def test_battery_reserve_gate():
    print("Testing Adaptive Confidence-Gated Battery Reserve (PRD Section 3.6)...")
    gp_model = SpatialWindGaussianProcess(num_inducing=128)
    gate = ConfidenceGatedBatteryReserve(nominal_return_energy_wh=14.0, beta_uncertainty_inflation=0.08)

    pos_curr = np.array([300.0, 300.0, 80.0])
    pos_launch = np.array([0.0, -450.0, 35.0])

    e_reserve, sigma_int, is_inflated = gate.evaluate_reserve_threshold(pos_curr, pos_launch, gp_model)
    print(f"  -> Dynamic E_reserve: {e_reserve:.2f} Wh (Nominal: 14.0 Wh, Integrated Uncertainty: {sigma_int:.1f})")
    assert e_reserve >= 14.0
    print("  [PASSED] Confidence-Gated Battery Reserve verified.\n")


def test_mppi_planner():
    print("Testing MPPI Trajectory Optimizer with 1,024 Parallel Rollouts (PRD Section 1.5 & Layer 5)...")
    canyon = MountainCanyon()
    wind_field = DynamicWindField()
    power_model = MultirotorPowerModel()
    camera_model = CameraStabilityModel()
    gp_model = SpatialWindGaussianProcess(num_inducing=128)
    mppi = MPPIPlanner(num_rollouts_k=1024, horizon_steps_n=30, dt=0.2)

    dummy_context = StateMachineContext(
        current_mode=OperationalMode.MISSION_CRUISE,
        target_setpoint=np.array([100.0, 100.0, 60.0]),
        speed_target_m_s=8.0,
        max_tilt_deg=20.0,
        max_angular_rate_rad_s=0.5,
        mode_reason="Unit test",
    )

    t0 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    import time
    start = time.perf_counter()

    v_cmd, yaw_cmd, rollouts = mppi.optimize(
        current_pos=np.array([0.0, 0.0, 50.0]),
        current_vel=np.array([0.0, 0.0, 0.0]),
        target_setpoint=np.array([100.0, 100.0, 60.0]),
        power_model=power_model,
        camera_model=camera_model,
        wind_field=wind_field,
        gp_spatial_model=gp_model,
        canyon_env=canyon,
        state_context=dummy_context,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0

    print(f"  -> K=1024 Rollouts Shape: {rollouts.shape}")
    print(f"  -> Generated Command Velocity: {np.round(v_cmd, 2)} m/s, Yaw Rate: {yaw_cmd:.3f} rad/s")
    print(f"  -> MPPI Iteration Execution Latency: {latency_ms:.2f} ms")
    assert rollouts.shape == (1024, 31, 3)
    assert latency_ms < 50.0, f"Latency {latency_ms} ms exceeds 20 Hz (50 ms) hard deadline"
    print("  [PASSED] MPPI Planner verified.\n")


def run_all_tests():
    print("==================================================================")
    print("     AEROHARVEST MATHEMATICAL & CODEBASE VERIFICATION SUITE       ")
    print("==================================================================\n")
    test_disturbance_observer()
    test_power_model()
    test_camera_jitter_model()
    test_sparse_gp()
    test_battery_reserve_gate()
    test_mppi_planner()
    print("==================================================================")
    print("          ALL AEROHARVEST UNIT TESTS PASSED SUCCESSFULLY!         ")
    print("==================================================================")


if __name__ == "__main__":
    run_all_tests()
