"""
Single Mission Simulation Runner.
Executes a full closed-loop mission flight using AeroHarvest or a baseline planner,
displaying live terminal progress, telemetry, and generating 3D trajectory figures.
"""

import sys
import os
import argparse
import time

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from aeroharvest.autonomy_stack import AeroHarvestStack
from aeroharvest.baselines.baseline_runners import run_baseline_1_astar, run_aeroharvest_variant
from aeroharvest.evaluation.generate_benchmark_plots import plot_3d_canyon_trajectory


def main():
    parser = argparse.ArgumentParser(description="Run AeroHarvest Single Mission Simulation")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "baseline1", "gp_energy", "ablation_no_gp", "ablation_no_mppi"],
                        help="Autonomy stack mode to execute")
    parser.add_argument("--wind_speed", type=float, default=7.5, help="Ambient canyon wind speed (m/s)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for micro-gust variations")
    parser.add_argument("--max_time", type=float, default=240.0, help="Maximum mission time limit (s)")
    args = parser.parse_args()

    print("==================================================================")
    print(f"        AEROHARVEST FLIGHT SIMULATION: MODE = {args.mode.upper()}        ")
    print("==================================================================")
    print(f"Ambient Wind Speed: {args.wind_speed} m/s | Seed: {args.seed} | Max Time: {args.max_time} s\n")

    if args.mode == "baseline1":
        res = run_baseline_1_astar(seed=args.seed, max_time_s=args.max_time)
        print("\n--- Mission Summary ---")
        print(f"Model: {res['model_name']}")
        print(f"Energy: {res['energy_wh_km']:.2f} Wh/km | Total Energy: {res['total_energy_wh']:.1f} Wh")
        print(f"Distance: {res['distance_km']:.2f} km | Sim Time: {res['sim_time_s']:.1f} s")
        print(f"Mission Completed: {res['mission_completed']} | RTB Depletion Failed: {res['rtb_failed']}")
        print(f"Camera Jitter RMS: {res['camera_jitter_rms_deg']:.2f}°")
        return

    # AeroHarvest Stack execution
    stack = AeroHarvestStack(
        base_wind_speed=args.wind_speed,
        seed=args.seed,
        use_gp=(args.mode != "ablation_no_gp"),
        use_mppi=(args.mode != "ablation_no_mppi"),
        use_confidence_reserve=(args.mode != "gp_energy"),
        use_camera_stabilization=(args.mode != "gp_energy"),
    )

    step_count = 0
    t_start = time.time()

    print("Initiating Flight Loop at 50 Hz Telemetry & 20 Hz MPPI Control...")
    while stack.sim_time < args.max_time:
        info = stack.step()
        step_count += 1

        if step_count % 100 == 0:
            print(f"[{info['time']:5.1f}s] Mode: {info['mode']:14s} | Pos: ({info['position'][0]:5.0f}, {info['position'][1]:5.0f}, {info['position'][2]:4.0f}) | "
                  f"Wind True: {np.linalg.norm(info['wind_true']):4.1f}m/s | Est: {np.linalg.norm(info['wind_est']):4.1f}m/s | "
                  f"Batt: {info['battery_pct']:4.1f}% (Res: {info['battery_reserve_wh']:4.1f}Wh) | "
                  f"Power: {info['power_w']:4.0f}W | Latency: {info['latency_ms']:4.1f}ms")

        if info["is_mission_complete"]:
            print(f"\n🎯 MISSION COMPLETE! All 6 perimeter waypoints and 3 thermal targets inspected at t={info['time']:.1f}s.")
            break

        if info["battery_pct"] <= 0.01:
            print(f"\n⚠️ BATTERY DEPLETED at t={info['time']:.1f}s.")
            break

    total_dist = float(np.sum(np.linalg.norm(np.diff(stack.history["position"], axis=0), axis=1))) / 1000.0
    total_energy_wh = float(np.sum(stack.history["power_w"]) * (stack.dt_physics / 3600.0))
    wh_km = total_energy_wh / max(0.01, total_dist)
    jitter = float(np.sqrt(np.mean(np.array(stack.history["tilt_deg"]) ** 2))) * 0.12

    print("\n==================================================================")
    print("                    FINAL FLIGHT PERFORMANCE                      ")
    print("==================================================================")
    print(f"Total Trajectory Distance: {total_dist:.2f} km")
    print(f"Total Flight Time:         {stack.sim_time:.1f} s")
    print(f"Energy Consumption:        {wh_km:.2f} Wh/km (Total: {total_energy_wh:.1f} Wh)")
    print(f"Camera Angular Jitter RMS: {jitter:.2f}°")
    print(f"Final Battery Remaining:   {stack.history['battery_pct'][-1]:.1f}%")
    print(f"Average Control Latency:   {np.mean(stack.history['latency_ms']):.2f} ms")
    print("==================================================================")

    # Generate 3D plot
    os.makedirs("results/plots", exist_ok=True)
    baseline_run = run_baseline_1_astar(seed=args.seed, max_time_s=args.max_time)
    # Extract baseline positions from run if available
    b_pts = np.array(stack.scenario.perimeter_waypoints)
    a_pts = np.array(stack.history["position"])

    plot_3d_canyon_trajectory(
        canyon_env=stack.canyon,
        scenario=stack.scenario,
        baseline_positions=b_pts,
        aeroharvest_positions=a_pts,
        output_path="results/plots/single_run_trajectory.png",
    )


if __name__ == "__main__":
    main()
