"""
Automated 30-Run Monte Carlo Simulation Engine (PRD Section 6.1).
Executes randomized seed trials across all Baselines, Ablations, and AeroHarvest.
"""

from typing import Dict, Any, List
import sys
import numpy as np
import pandas as pd
from aeroharvest.baselines.baseline_runners import (
    run_baseline_1_astar,
    run_baseline_2_reactive,
    run_aeroharvest_variant,
)
from aeroharvest.evaluation.metrics_logger import BenchmarkMetricsLogger


class MonteCarloSimulationEngine:
    """
    Executes N-run Monte Carlo simulation suite across all models.
    """

    def __init__(self, num_runs: int = 30):
        self.num_runs = num_runs
        self.logger = BenchmarkMetricsLogger()

    def run_full_suite(self, max_time_per_run_s: float = 120.0, verbose: bool = True) -> pd.DataFrame:
        """
        Runs Monte Carlo trials for:
        1. Baseline 1 (A*)
        2. Baseline 2 (Reactive L1)
        3. Baseline 3 (GP-Energy)
        4. AeroHarvest (Full proposed)
        5. Ablation 1 (GP-Disabled)
        6. Ablation 2 (MPPI-Disabled)
        """
        models = [
            ("Baseline 1 (A*)", lambda s: run_baseline_1_astar(seed=s, max_time_s=max_time_per_run_s)),
            ("Baseline 2 (Reactive)", lambda s: run_baseline_2_reactive(seed=s, max_time_s=max_time_per_run_s)),
            ("Baseline 3 (GP-Energy)", lambda s: run_aeroharvest_variant(
                "Baseline 3 (GP-Energy)", seed=s, use_gp=True, use_mppi=True,
                use_confidence_reserve=False, use_camera_stabilization=False, max_time_s=max_time_per_run_s
            )),
            ("Ablation 1 (GP-Disabled)", lambda s: run_aeroharvest_variant(
                "Ablation 1 (GP-Disabled)", seed=s, use_gp=False, use_mppi=True,
                use_confidence_reserve=False, use_camera_stabilization=True, max_time_s=max_time_per_run_s
            )),
            ("Ablation 2 (MPPI-Disabled)", lambda s: run_aeroharvest_variant(
                "Ablation 2 (MPPI-Disabled)", seed=s, use_gp=True, use_mppi=False,
                use_confidence_reserve=True, use_camera_stabilization=True, max_time_s=max_time_per_run_s
            )),
            ("AeroHarvest (Full)", lambda s: run_aeroharvest_variant(
                "AeroHarvest (Full)", seed=s, use_gp=True, use_mppi=True,
                use_confidence_reserve=True, use_camera_stabilization=True, max_time_s=max_time_per_run_s
            )),
        ]

        if verbose:
            print(f"=== Starting {self.num_runs}-Run Monte Carlo Evaluation Suite ===")

        for model_name, runner_func in models:
            if verbose:
                print(f"\nEvaluating {model_name}...")
            for run_idx in range(self.num_runs):
                seed = 1000 + run_idx * 17
                result = runner_func(seed)
                self.logger.record_run(model_name, run_idx, result)
                if verbose and (run_idx + 1) % 5 == 0:
                    print(f"  -> Completed Run {run_idx + 1}/{self.num_runs} (Energy: {result['energy_wh_km']:.1f} Wh/km)")

        summary = self.logger.compute_summary_table()
        return summary
