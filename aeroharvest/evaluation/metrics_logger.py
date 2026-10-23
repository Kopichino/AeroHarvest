"""
Quantitative Metric Logger & Statistical Benchmark Aggregator (PRD Section 6.4).
Computes mean +/- std, failure rates, camera jitter RMS, and control latencies
matching the expected PRD quantitative benchmark table.
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd


class BenchmarkMetricsLogger:
    """
    Logs and calculates academic major project benchmark metrics.
    """

    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def record_run(self, model_name: str, run_id: int, metrics: Dict[str, Any]):
        rec = {
            "model_name": model_name,
            "run_id": run_id,
            "energy_wh_km": metrics["energy_wh_km"],
            "total_energy_wh": metrics["total_energy_wh"],
            "distance_km": metrics["distance_km"],
            "mission_completed": 1 if metrics["mission_completed"] else 0,
            "rtb_failed": 1 if metrics["rtb_failed"] else 0,
            "camera_jitter_rms_deg": metrics["camera_jitter_rms_deg"],
            "wind_rmse_m_s": metrics["wind_rmse_m_s"],
            "mean_latency_ms": metrics["mean_latency_ms"],
        }
        self.records.append(rec)

    def compute_summary_table(self) -> pd.DataFrame:
        """
        Generates aggregate statistical summary matching PRD Section 6.4 Table.
        """
        df = pd.DataFrame(self.records)
        if df.empty:
            return pd.DataFrame()

        summary_rows = []
        for model, group in df.groupby("model_name", sort=False):
            n_runs = len(group)
            e_mean, e_std = group["energy_wh_km"].mean(), group["energy_wh_km"].std()
            completion_rate = (group["mission_completed"].sum() / n_runs) * 100.0
            rtb_fail_rate = (group["rtb_failed"].sum() / n_runs) * 100.0
            jitter_mean = group["camera_jitter_rms_deg"].mean()
            wind_rmse_mean = group["wind_rmse_m_s"].mean()
            wind_rmse_std = group["wind_rmse_m_s"].std()
            latency_mean = group["mean_latency_ms"].mean()

            summary_rows.append({
                "Model / Architecture": model,
                "Mean Energy (Wh/km)": f"{e_mean:.1f} +/- {e_std:.1f}",
                "Mission Completion (%)": f"{completion_rate:.1f}%",
                "RTB Failure Rate (%)": f"{rtb_fail_rate:.1f}%",
                "Camera Jitter RMS (deg)": f"{jitter_mean:.2f} deg",
                "Wind Estimation RMSE (m/s)": f"{wind_rmse_mean:.2f} +/- {wind_rmse_std:.2f}",
                "Control Latency (ms)": f"{latency_mean:.1f} ms",
            })

        return pd.DataFrame(summary_rows)

    def export_csv(self, file_path: str):
        df = pd.DataFrame(self.records)
        df.to_csv(file_path, index=False)
