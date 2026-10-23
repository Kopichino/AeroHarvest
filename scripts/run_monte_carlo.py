"""
Automated 30-Run Monte Carlo Simulation Runner (PRD Section 6.1 & 6.4).
Executes Monte Carlo evaluation sweeps across:
1. Baseline 1 (A*)
2. Baseline 2 (Reactive L1)
3. Baseline 3 (GP-Energy)
4. Ablation 1 (GP-Disabled)
5. Ablation 2 (MPPI-Disabled)
6. AeroHarvest (Full Stack)
Saves tabular metrics to CSV and renders publication figures in results/plots/.
"""

import sys
import os
import argparse
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aeroharvest.evaluation.monte_carlo_runner import MonteCarloSimulationEngine
from aeroharvest.evaluation.generate_benchmark_plots import generate_benchmark_figures


def main():
    parser = argparse.ArgumentParser(description="AeroHarvest Monte Carlo Simulation Suite")
    parser.add_argument("--runs", type=int, default=30, help="Number of Monte Carlo runs per architecture (PRD default: 30)")
    parser.add_argument("--time_limit", type=float, default=90.0, help="Simulated flight time limit per trial (s)")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save CSV logs and plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("==================================================================")
    print(f"      AEROHARVEST {args.runs}-RUN MONTE CARLO EVALUATION BENCHMARK     ")
    print("==================================================================")
    print(f"Total Trials per Architecture: {args.runs}")
    print(f"Simulation Time Limit:        {args.time_limit} s")
    print(f"Results Output Directory:     {args.output_dir}/\n")

    engine = MonteCarloSimulationEngine(num_runs=args.runs)
    summary_df = engine.run_full_suite(max_time_per_run_s=args.time_limit, verbose=True)

    # Save detailed CSV
    csv_path = os.path.join(args.output_dir, "monte_carlo_results.csv")
    engine.logger.export_csv(csv_path)
    print(f"\n[OK] Raw run records exported to {csv_path}")

    # Generate publication figures
    raw_df = pd.DataFrame(engine.logger.records)
    generate_benchmark_figures(raw_df, output_dir=plots_dir)

    print("\n==================================================================")
    print("       AEROHARVEST QUANTITATIVE BENCHMARK EVALUATION SUMMARY      ")
    print("==================================================================")
    print(summary_df.to_string(index=False))
    print("==================================================================")


if __name__ == "__main__":
    main()
