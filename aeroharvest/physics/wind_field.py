"""
3D Spatial Wind Vector Field & Micro-Gust Generator.
Models spatial orographic wind flows in mountain valleys, canyon channeling drafts,
leeward recirculation, and Gaussian micro-gust bursts (0-14 m/s, 3.0 m/s peak gusts).
"""

from typing import Tuple, Union
import numpy as np
import torch


class DynamicWindField:
    """
    Spatially varying 3D wind field v_wind(x, y, z, t) in the inertial frame.
    """

    def __init__(
        self,
        base_wind_speed: float = 6.0,
        base_wind_heading_deg: float = 45.0,
        gust_amplitude: float = 3.0,
        gust_frequency_hz: float = 0.8,
        seed: int | None = 42,
    ):
        self.rng = np.random.default_rng(seed)
        self.base_speed = base_wind_speed
        heading_rad = np.radians(base_wind_heading_deg)
        self.base_dir = np.array([np.cos(heading_rad), np.sin(heading_rad), 0.0])

        self.gust_amplitude = gust_amplitude
        self.gust_freq = gust_frequency_hz

        # Randomized phase offsets for spatial turbulent Fourier components
        self.k_vecs = np.array([
            [0.012, 0.008, 0.015],
            [-0.015, 0.010, 0.020],
            [0.006, -0.018, 0.010],
            [0.025, 0.020, 0.005],
        ])
        self.spatial_phases = self.rng.uniform(0, 2 * np.pi, size=(4, 3))
        self.spatial_weights = np.array([1.8, 1.2, 0.9, 0.5])

        # Canyon jet channel center line: y = 80 sin(0.004 x)
        self.canyon_jet_speed_boost = 4.5  # m/s channel acceleration (Venturi effect)

    def get_wind_at(self, position: np.ndarray, t: float = 0.0) -> np.ndarray:
        """
        Returns true 3D wind vector [v_wx, v_wy, v_wz] in m/s at position [x, y, z] and time t.
        """
        pos = np.asarray(position, dtype=np.float64)
        x, y, z = pos[0], pos[1], pos[2]

        # 1. Base synoptic wind (height-dependent boundary layer gradient)
        # Logarithmic/power law altitude factor
        z_norm = np.clip(z / 150.0, 0.2, 2.5)
        v_base = self.base_speed * (z_norm ** 0.25) * self.base_dir

        # 2. Canyon Channeling Draft (Venturi jet along trench floor)
        y_center = 80.0 * np.sin(0.004 * x)
        dist_canyon = np.abs(y - y_center)
        canyon_factor = np.exp(-((dist_canyon / 90.0) ** 2)) * np.exp(-((z / 120.0) ** 2))
        v_channel = canyon_factor * self.canyon_jet_speed_boost * np.array([0.85, 0.52, 0.1])

        # 3. Orographic vertical draft (upward along windward slopes, downdraft along leeward)
        slope_x = -0.05 * np.sin(0.01 * x)
        slope_y = 0.06 * np.cos(0.01 * y)
        w_updraft = 1.8 * (slope_x * self.base_dir[0] + slope_y * self.base_dir[1])
        v_oro = np.array([0.0, 0.0, w_updraft])

        # 4. Spatially coherent turbulence harmonics
        v_turb = np.zeros(3)
        for i in range(len(self.spatial_weights)):
            phase = np.dot(self.k_vecs[i], pos) + self.spatial_phases[i]
            v_turb += self.spatial_weights[i] * np.sin(phase)

        # 5. Temporal micro-gust bursts (Gaussian pulse train)
        gust_mod = np.sin(2 * np.pi * self.gust_freq * t) + 0.5 * np.cos(2 * np.pi * 0.35 * t + 1.2)
        v_gust = self.gust_amplitude * gust_mod * np.array([
            np.cos(0.01 * x + t),
            np.sin(0.01 * y + t),
            0.4 * np.sin(2 * np.pi * self.gust_freq * t),
        ])

        v_wind = v_base + v_channel + v_oro + v_turb + v_gust
        return v_wind

    def get_wind_derivative(self, position: np.ndarray, t: float, dt: float = 0.05) -> np.ndarray:
        """Computes d(v_wind)/dt for camera gust-rate penalty."""
        w1 = self.get_wind_at(position, t)
        w2 = self.get_wind_at(position, t + dt)
        return (w2 - w1) / dt

    def get_wind_batch_torch(self, positions_torch: torch.Tensor, t: float = 0.0) -> torch.Tensor:
        """
        Evaluates wind field vector [vx, vy, vz] for a batch of 3D positions in PyTorch.
        positions_torch: Tensor of shape (K, N, 3) or (B, 3).
        Returns: Tensor of same leading shape with 3 wind components.
        """
        device = positions_torch.device
        dtype = positions_torch.dtype

        x = positions_torch[..., 0]
        y = positions_torch[..., 1]
        z = positions_torch[..., 2]

        z_norm = torch.clamp(z / 150.0, 0.2, 2.5)
        base_dir_t = torch.tensor(self.base_dir, device=device, dtype=dtype)
        v_base = (self.base_speed * (z_norm ** 0.25)).unsqueeze(-1) * base_dir_t

        y_center = 80.0 * torch.sin(0.004 * x)
        dist_canyon = torch.abs(y - y_center)
        canyon_factor = torch.exp(-((dist_canyon / 90.0) ** 2)) * torch.exp(-((z / 120.0) ** 2))
        channel_dir_t = torch.tensor([0.85, 0.52, 0.1], device=device, dtype=dtype)
        v_channel = (canyon_factor * self.canyon_jet_speed_boost).unsqueeze(-1) * channel_dir_t

        # Harmonic spatial turbulence
        v_turb_x = 1.5 * torch.sin(0.012 * x + 0.008 * y + 0.5)
        v_turb_y = 1.5 * torch.cos(0.015 * x - 0.010 * y + 1.2)
        v_turb_z = 0.8 * torch.sin(0.010 * z + 0.006 * x)
        v_turb = torch.stack([v_turb_x, v_turb_y, v_turb_z], dim=-1)

        gust_mod = float(np.sin(2 * np.pi * self.gust_freq * t))
        v_gust = torch.tensor([
            self.gust_amplitude * gust_mod * 0.7,
            self.gust_amplitude * gust_mod * 0.7,
            self.gust_amplitude * gust_mod * 0.3,
        ], device=device, dtype=dtype)

        v_wind = v_base + v_channel + v_turb + v_gust
        return v_wind
