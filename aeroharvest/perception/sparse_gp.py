"""
GPU-Accelerated Sparse Gaussian Process (GPyTorch / PyTorch) with Matérn 5/2 Covariance.
Maintains an online 3D spatial wind belief map (mean mu_w, predictive variance sigma_w^2)
using M=128 inducing points sliding window over mountain canyon airspace.
Mathematical reference: PRD Section 3.2 (Equations 128 - 153).
"""

from typing import Tuple, List, Optional
import numpy as np
import torch
try:
    import gpytorch
except ImportError:
    gpytorch = None


class SpatialWindGaussianProcess:
    """
    3D Spatial Wind Belief Engine managing 3 independent GPs for [v_wx, v_wy, v_wz].
    Supports batch querying of mean and variance over trajectory rollouts and voxel grids.
    Mathematical formulation: PRD Section 3.2 (Equations 128 - 153).
    """

    def __init__(
        self,
        num_inducing: int = 128,
        lengthscale: float = 40.0,
        signal_variance: float = 2.0,
        observation_noise: float = 0.04,
        max_buffer_size: int = 500,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.num_inducing = num_inducing
        self.lengthscale = lengthscale
        self.signal_variance = signal_variance
        self.observation_noise = observation_noise
        self.max_buffer_size = max_buffer_size

        # Circular observation buffer: positions (N, 3), observations (N, 3)
        self.obs_positions: List[np.ndarray] = []
        self.obs_wind: List[np.ndarray] = []

        # Default inducing points distributed across canyon volume
        self.inducing_points = self._initialize_inducing_grid()

        # Cached kernel inverse and weights for M inducing landmarks
        self.K_uu_inv = None
        self.alpha_weights = None  # (M, 3) weights for mean prediction

        # Prior uncertainty at unvisited regions
        self.prior_variance = float(self.signal_variance + self.observation_noise)

    def _initialize_inducing_grid(self) -> torch.Tensor:
        """Initializes M=128 inducing points covering nominal canyon workspace."""
        nx, ny, nz = 6, 6, 4  # 6*6*4 = 144 -> sliced to M=128
        xs = np.linspace(-450.0, 450.0, nx)
        ys = np.linspace(-450.0, 450.0, ny)
        zs = np.linspace(30.0, 250.0, nz)
        grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs)
        pts = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
        selected = pts[: self.num_inducing]
        return torch.tensor(selected, dtype=torch.float32, device=self.device)

    def matern_52_kernel_torch(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Matérn 5/2 Covariance Kernel (PRD Equation 134):
        k(x, x') = sigma_f^2 * (1 + sqrt(5)*d/l + 5*d^2/(3*l^2)) * exp(-sqrt(5)*d/l)
        """
        # Pairwise distance matrix
        dists = torch.cdist(x1, x2, p=2.0)
        sqrt5_d_l = np.sqrt(5.0) * dists / self.lengthscale
        term = 1.0 + sqrt5_d_l + (5.0 * dists ** 2) / (3.0 * (self.lengthscale ** 2))
        cov = self.signal_variance * term * torch.exp(-sqrt5_d_l)
        return cov

    def add_observation(self, position: np.ndarray, wind_vector: np.ndarray):
        """Adds a new (pos, wind) observation from the disturbance observer."""
        self.obs_positions.append(np.asarray(position, dtype=np.float32))
        self.obs_wind.append(np.asarray(wind_vector, dtype=np.float32))

        if len(self.obs_positions) > self.max_buffer_size:
            self.obs_positions.pop(0)
            self.obs_wind.pop(0)

    def fit_online(self, steps: int = 1):
        """
        Updates the inducing landmarks and evaluates the analytical Sparse GP
        variational conditioning matrices (PRD Section 3.2, Equations 148-153).
        """
        n_obs = len(self.obs_positions)
        if n_obs < 3:
            return

        obs_x = torch.tensor(np.array(self.obs_positions), dtype=torch.float32, device=self.device)
        obs_y = torch.tensor(np.array(self.obs_wind), dtype=torch.float32, device=self.device)

        # Filter observations to keep spatially distinct landmarks (minimum 3m separation)
        unique_pts = []
        unique_y = []
        for i in range(n_obs):
            p = obs_x[i]
            if len(unique_pts) == 0:
                unique_pts.append(p)
                unique_y.append(obs_y[i])
            else:
                dists = torch.norm(torch.stack(unique_pts) - p, dim=-1)
                if torch.min(dists) > 3.0:
                    unique_pts.append(p)
                    unique_y.append(obs_y[i])

        if len(unique_pts) < 3:
            return

        ind_x = torch.stack(unique_pts[-self.num_inducing:])
        ind_y = torch.stack(unique_y[-self.num_inducing:])
        self.inducing_points = ind_x
        M = ind_x.size(0)

        # Compute K_uu
        K_uu = self.matern_52_kernel_torch(ind_x, ind_x)
        reg_eye = torch.eye(M, device=self.device)
        K_uu_reg = K_uu + 0.05 * reg_eye

        # Cross-covariance between inducing points and observations: K_ux (M, N)
        K_ux = self.matern_52_kernel_torch(ind_x, obs_x)

        inv_noise = 1.0 / self.observation_noise
        lambda_mat = K_uu_reg + inv_noise * (K_ux @ K_ux.T) + 0.05 * reg_eye
        rhs = inv_noise * (K_ux @ obs_y)  # (M, 3)

        try:
            L = torch.linalg.cholesky(lambda_mat)
            self.lambda_inv = torch.cholesky_inverse(L)
            self.alpha_weights = self.lambda_inv @ rhs
            self.K_uu_inv = torch.cholesky_inverse(torch.linalg.cholesky(K_uu_reg))
        except (RuntimeError, torch._C._LinAlgError):
            self.lambda_inv = torch.linalg.pinv(lambda_mat)
            self.alpha_weights = self.lambda_inv @ rhs
            self.K_uu_inv = torch.linalg.pinv(K_uu_reg)

    def predict(
        self, query_positions: np.ndarray | torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Queries the GP for predictive mean mu_w and variance sigma_w^2 (PRD Eq 148-153).
        Returns:
            mean: (N, 3) 3D wind velocity vector mean
            variance: (N,) total spatial uncertainty variance
        """
        if isinstance(query_positions, np.ndarray):
            q_tensor = torch.tensor(query_positions, dtype=torch.float32, device=self.device)
        else:
            q_tensor = query_positions.to(self.device).float()

        if q_tensor.ndim == 1:
            q_tensor = q_tensor.unsqueeze(0)

        N = q_tensor.size(0)

        if self.alpha_weights is None or len(self.obs_positions) < 3:
            zeros = np.zeros((N, 3), dtype=np.float64)
            prior_vars = np.full(N, self.prior_variance, dtype=np.float64)
            return zeros, prior_vars

        with torch.no_grad():
            # Cross-covariance k_*u: (N, M)
            k_star_u = self.matern_52_kernel_torch(q_tensor, self.inducing_points)

            # Predictive mean: mu_w = k_*u * alpha_weights (PRD Eq 148)
            mu = k_star_u @ self.alpha_weights  # (N, 3)

            # Predictive variance: sigma^2(x_*) = k(x_*, x_*) - k_*u * (K_uu^-1 - lambda^-1) * k_u*
            # PRD Eq 152: variance reduction from observations
            v_reduct = torch.sum(k_star_u * (k_star_u @ (self.K_uu_inv - self.lambda_inv)), dim=-1)
            var_pred = torch.clamp(self.signal_variance - v_reduct + self.observation_noise, min=0.01)

        return mu.cpu().numpy(), var_pred.cpu().numpy()

    def predict_torch(self, query_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native batch query for MPPI rollouts."""
        q = query_tensor.to(self.device).float()
        orig_shape = q.shape[:-1]
        q_flat = q.reshape(-1, 3)
        N = q_flat.size(0)

        if self.alpha_weights is None or len(self.obs_positions) < 3:
            mean_flat = torch.zeros((N, 3), device=self.device)
            var_flat = torch.full((N,), self.prior_variance, device=self.device)
            return mean_flat.reshape(*orig_shape, 3), var_flat.reshape(*orig_shape)

        with torch.no_grad():
            k_star_u = self.matern_52_kernel_torch(q_flat, self.inducing_points)
            mu_flat = k_star_u @ self.alpha_weights
            v_reduct = torch.sum(k_star_u * (k_star_u @ (self.K_uu_inv - self.lambda_inv)), dim=-1)
            var_flat = torch.clamp(self.signal_variance - v_reduct + self.observation_noise, min=0.01)

        return mu_flat.reshape(*orig_shape, 3), var_flat.reshape(*orig_shape)
