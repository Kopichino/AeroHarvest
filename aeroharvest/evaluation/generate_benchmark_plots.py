"""
Publication-Quality Benchmark Plotting & Visualization Suite.
Generates academic figures for Major Project Report and Committee Review Presentation:
1. Energy Consumption (Wh/km) Comparison
2. Camera Angular Jitter RMS (deg)
3. Return-to-Base Failure Rate & Mission Completion
4. 3D Trajectory in Mountain Canyon
5. Multi-Panel Performance Dashboard
"""

import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def generate_benchmark_figures(results_df: pd.DataFrame, output_dir: str = "results/plots"):
    """
    Renders publication-ready charts from Monte Carlo results.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    models = [
        "Baseline 1 (A*)",
        "Baseline 2 (Reactive)",
        "Baseline 3 (GP-Energy)",
        "Ablation 1 (GP-Disabled)",
        "Ablation 2 (MPPI-Disabled)",
        "AeroHarvest (Full)",
    ]

    # Clean data from dataframe
    df_filtered = results_df[results_df["model_name"].isin(models)]

    # 1. Energy Consumption Plot
    plt.figure(figsize=(9, 5), dpi=300)
    energy_means = df_filtered.groupby("model_name", sort=False)["energy_wh_km"].mean()
    energy_stds = df_filtered.groupby("model_name", sort=False)["energy_wh_km"].std()

    colors = ["#e74c3c", "#e67e22", "#f39c12", "#95a5a6", "#3498db", "#2ecc71"]
    bars = plt.bar(energy_means.index, energy_means.values, yerr=energy_stds.values, capsize=5, color=colors, alpha=0.9, edgecolor="black")
    plt.ylabel("Mean Energy Consumption (Wh/km)", fontsize=12, fontweight="bold")
    plt.title("Energy Consumption Benchmark Across Architectures (Wh/km)", fontsize=13, fontweight="bold")
    plt.xticks(rotation=20, ha="right", fontsize=10)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.0, f"{yval:.1f}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "benchmark_energy_consumption.png"))
    plt.close()

    # 2. Camera Angular Jitter RMS Plot
    plt.figure(figsize=(9, 5), dpi=300)
    jitter_means = df_filtered.groupby("model_name", sort=False)["camera_jitter_rms_deg"].mean()
    bars = plt.bar(jitter_means.index, jitter_means.values, color=colors, alpha=0.9, edgecolor="black")
    plt.ylabel("Camera Angular Jitter RMS (deg)", fontsize=12, fontweight="bold")
    plt.title("Camera Payload Vibration & Jitter RMS (Lower is Better)", fontsize=13, fontweight="bold")
    plt.axhline(y=1.8, color="#27ae60", linestyle="--", label="Target Jitter (< 2.0 deg)")
    plt.xticks(rotation=20, ha="right", fontsize=10)
    plt.legend()
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.2, f"{yval:.1f}°", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "benchmark_camera_jitter.png"))
    plt.close()

    # 3. Mission Completion & RTB Failure Rates
    plt.figure(figsize=(9, 5), dpi=300)
    completion = df_filtered.groupby("model_name", sort=False)["mission_completed"].mean() * 100.0
    rtb_fails = df_filtered.groupby("model_name", sort=False)["rtb_failed"].mean() * 100.0

    x = np.arange(len(completion))
    width = 0.35
    plt.bar(x - width / 2, completion.values, width, label="Mission Completion (%)", color="#2980b9", edgecolor="black")
    plt.bar(x + width / 2, rtb_fails.values, width, label="RTB Depletion Failure (%)", color="#c0392b", edgecolor="black")
    plt.xticks(x, completion.index, rotation=20, ha="right", fontsize=10)
    plt.ylabel("Percentage (%)", fontsize=12, fontweight="bold")
    plt.title("Mission Completion vs. Return-to-Base Failure Rate", fontsize=13, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "benchmark_rtb_failures.png"))
    plt.close()

    # 4. Multi-Panel Comprehensive Performance Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # (a) Energy
    axes[0, 0].bar(energy_means.index, energy_means.values, yerr=energy_stds.values, capsize=4, color=colors, edgecolor="black")
    axes[0, 0].set_title("A. Energy Efficiency (Wh/km)", fontweight="bold")
    axes[0, 0].set_ylabel("Wh/km")
    axes[0, 0].tick_params(axis="x", labelrotation=25)

    # (b) Jitter
    axes[0, 1].bar(jitter_means.index, jitter_means.values, color=colors, edgecolor="black")
    axes[0, 1].set_title("B. Camera Payload Angular Jitter (deg)", fontweight="bold")
    axes[0, 1].set_ylabel("RMS Deg")
    axes[0, 1].tick_params(axis="x", labelrotation=25)

    # (c) Completion
    axes[1, 0].bar(completion.index, completion.values, color="#3498db", edgecolor="black")
    axes[1, 0].set_title("C. Mission Completion Rate (%)", fontweight="bold")
    axes[1, 0].set_ylabel("Percentage")
    axes[1, 0].set_ylim(0, 105)
    axes[1, 0].tick_params(axis="x", labelrotation=25)

    # (d) Control Latency
    latency_means = df_filtered.groupby("model_name", sort=False)["mean_latency_ms"].mean()
    axes[1, 1].bar(latency_means.index, latency_means.values, color="#9b59b6", edgecolor="black")
    axes[1, 1].set_title("D. Control Loop Latency (ms)", fontweight="bold")
    axes[1, 1].set_ylabel("Latency (ms)")
    axes[1, 1].axhline(y=50.0, color="red", linestyle="--", label="20 Hz Hard Deadline (50 ms)")
    axes[1, 1].legend()
    axes[1, 1].tick_params(axis="x", labelrotation=25)

    plt.suptitle("AeroHarvest Capstone Evaluation: Monte Carlo Performance Benchmarks", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "benchmark_comprehensive_dashboard.png"))
    plt.close()

    print(f"Publication-quality figures successfully saved to {output_dir}/")


def plot_3d_canyon_trajectory(
    canyon_env,
    scenario,
    baseline_positions: np.ndarray,
    aeroharvest_positions: np.ndarray,
    output_path: str = "results/plots/trajectory_3d_comparison.png",
):
    """
    Renders 3D trajectory comparison through Mountain Canyon topography.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig = plt.figure(figsize=(12, 8), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    # Canyon terrain grid
    xs = np.linspace(-450, 450, 60)
    ys = np.linspace(-450, 450, 60)
    X, Y = np.meshgrid(xs, ys)
    Z = canyon_env.get_elevation(X, Y)

    # Plot terrain surface
    ax.plot_surface(X, Y, Z, cmap="terrain", alpha=0.45, edgecolor="none")

    # Plot waypoints
    wps = np.array(scenario.perimeter_waypoints)
    ax.scatter(wps[:, 0], wps[:, 1], wps[:, 2], color="black", s=60, marker="o", label="Perimeter Waypoints")

    # Plot thermal targets
    for target in scenario.thermal_targets:
        ax.scatter(target.position[0], target.position[1], target.position[2], color="red", s=120, marker="^", label=f"Thermal Target {target.target_id}")

    # Plot Baseline 1 Trajectory
    if len(baseline_positions) > 0:
        ax.plot(
            baseline_positions[:, 0],
            baseline_positions[:, 1],
            baseline_positions[:, 2],
            color="#e74c3c",
            linestyle="--",
            linewidth=2.2,
            label="Baseline 1 (A* Geometric - Depleted)",
        )

    # Plot AeroHarvest Trajectory
    if len(aeroharvest_positions) > 0:
        ax.plot(
            aeroharvest_positions[:, 0],
            aeroharvest_positions[:, 1],
            aeroharvest_positions[:, 2],
            color="#27ae60",
            linestyle="-",
            linewidth=2.5,
            label="AeroHarvest (Wind-Adaptive + Probe)",
        )

    ax.set_xlabel("X (m)", fontweight="bold")
    ax.set_ylabel("Y (m)", fontweight="bold")
    ax.set_zlabel("Altitude Z (m)", fontweight="bold")
    ax.set_title("3D Mountain Canyon Trajectory: AeroHarvest vs. Baseline 1", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"3D Canyon trajectory plot saved to {output_path}")
