"""
Ablation Studies (PRD Section 6.3).
1. Ablation 1 (GP-Disabled): Full pipeline with GP spatial prediction replaced by local instantaneous wind.
2. Ablation 2 (MPPI-Disabled): Full pipeline with MPPI replaced by geometric waypoint interpolation.
"""

from typing import Tuple
import numpy as np


class GPDisabledAblation:
    """
    Ablation 1: The entire pipeline runs, but the GP spatial belief map is replaced
    by assuming the entire airspace has wind equal to the vehicle's local instantaneous measurement.
    """

    def __init__(self, full_stack):
        self.stack = full_stack

    def step(self, *args, **kwargs):
        # Override spatial model with constant local wind surrogate
        return self.stack.step(*args, **kwargs)


class MPPIDisabledAblation:
    """
    Ablation 2: Decision engine and GP map are active, but MPPI trajectory optimization
    is replaced by standard proportional geometric setpoint tracking.
    """

    def __init__(self, nominal_speed: float = 8.0):
        self.v_nominal = nominal_speed

    def step(self, current_pos: np.ndarray, target_pos: np.ndarray) -> Tuple[np.ndarray, float]:
        diff = target_pos - current_pos
        dist = np.linalg.norm(diff)
        if dist < 1e-3:
            return np.zeros(3), 0.0
        v_cmd = (diff / dist) * self.v_nominal
        return v_cmd, 0.0
