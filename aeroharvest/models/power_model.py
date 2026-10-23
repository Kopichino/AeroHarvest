"""
Multirotor Electrical Power Model (PRD Section 3.3).
Calculates instant electrical power consumption accounting for momentum-theory induced hover,
parasite drag, and avionics base draw.
"""

from typing import Union
import numpy as np
import torch


class MultirotorPowerModel:
    """
    Mathematical Implementation of PRD Equation 160:
    P_elec(v_rel) = P_hover / sqrt( sqrt(1 + (v_rel/(2*vi))^4) - (v_rel/(2*vi))^2 )
                    + 0.5 * rho * Cd * A * ||v_rel||^3 + P_electronics
    """

    def __init__(
        self,
        p_hover_w: float = 380.0,
        v_induced_m_s: float = 5.2,
        p_electronics_w: float = 25.0,
        air_density_kg_m3: float = 1.225,
        drag_coeff_cd: float = 0.45,
        projected_frontal_area_m2: float = 0.12,
    ):
        self.p_hover = p_hover_w
        self.v_i = v_induced_m_s
        self.p_elec_base = p_electronics_w
        self.rho = air_density_kg_m3
        self.cd = drag_coeff_cd
        self.area = projected_frontal_area_m2

        self.drag_factor_cube = 0.5 * self.rho * self.cd * self.area

    def compute_power_np(self, v_rel: np.ndarray) -> float:
        """
        Computes electrical power in Watts for a single relative velocity vector or scalar speed.
        v_rel: 3D vector [u, v, w] or scalar airspeed in m/s.
        """
        if isinstance(v_rel, np.ndarray) and v_rel.ndim > 0:
            speed_rel = float(np.linalg.norm(v_rel))
        else:
            speed_rel = float(v_rel)

        # Dimensionless ratio: v_rel / (2 * v_i)
        ratio = speed_rel / (2.0 * self.v_i)
        ratio_sq = ratio * ratio
        ratio_quad = ratio_sq * ratio_sq

        # Momentum theory induced power factor
        induced_denom = np.sqrt(np.sqrt(1.0 + ratio_quad) - ratio_sq)
        induced_denom = max(induced_denom, 1e-4)

        p_induced_profile = self.p_hover / induced_denom

        # Parasite drag power: 0.5 * rho * Cd * A * ||v_rel||^3
        p_parasite = self.drag_factor_cube * (speed_rel ** 3)

        p_total = p_induced_profile + p_parasite + self.p_elec_base
        return p_total

    def compute_power_torch(self, v_rel_tensor: torch.Tensor) -> torch.Tensor:
        """
        Batch evaluation of power in Watts using PyTorch for MPPI trajectory rollouts.
        v_rel_tensor: Tensor of shape (..., 3) or (..., 1) or (...).
        """
        if v_rel_tensor.shape[-1] == 3:
            speed_rel = torch.norm(v_rel_tensor, dim=-1)
        else:
            speed_rel = v_rel_tensor

        ratio = speed_rel / (2.0 * self.v_i)
        ratio_sq = ratio * ratio
        ratio_quad = ratio_sq * ratio_sq

        # Induced term
        induced_denom = torch.sqrt(torch.clamp(torch.sqrt(1.0 + ratio_quad) - ratio_sq, min=1e-6))
        p_induced_profile = self.p_hover / induced_denom

        # Parasite term
        p_parasite = self.drag_factor_cube * torch.pow(speed_rel, 3)

        p_total = p_induced_profile + p_parasite + self.p_elec_base
        return p_total
