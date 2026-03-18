"""
Physics-Informed Neural Network (PINN) solver for skin transport.

Uses deepxde-style architecture (参考代码/具体代码/deepxde-master/) but
reimplemented in pure PyTorch for tighter integration with the state
modulation network.

The PINN enforces the layered diffusion PDE as a soft constraint via
residual loss, combined with boundary/initial condition losses and
(critically) data-fitting loss from IVPT observations.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class SkinPINNNetwork(nn.Module):
    """Neural network approximating C(x, t) for the skin diffusion problem.

    Architecture mirrors deepxde's feedforward NN with Fourier feature input
    layer for better resolution of sharp gradients at layer interfaces.
    """

    def __init__(
        self,
        hidden_layers: List[int] = None,
        activation: str = "tanh",
        fourier_features: int = 32,
    ) -> None:
        super().__init__()
        hidden_layers = hidden_layers or [128, 128, 128, 128]

        # Fourier feature embedding for (x, t)
        self.ff_B = nn.Parameter(
            torch.randn(2, fourier_features) * 2.0, requires_grad=False
        )
        in_dim = 2 * fourier_features + 2  # sin + cos + raw (x, t)

        layers: List[nn.Module] = []
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "silu":
                layers.append(nn.SiLU())
            else:
                layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))  # output: C(x, t)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, 1) spatial coordinate
            t: (N, 1) time coordinate
        Returns:
            C: (N, 1) predicted concentration
        """
        xt = torch.cat([x, t], dim=-1)
        proj = xt @ self.ff_B
        ff = torch.cat([torch.sin(proj), torch.cos(proj), xt], dim=-1)
        return self.net(ff)


class SkinPINNSolver:
    """Train a PINN for the skin layered diffusion PDE.

    Loss = w_pde · L_pde + w_bc · L_bc + w_ic · L_ic + w_data · L_data

    The PDE residual adapts to spatially-varying D(x) and k(x), which are
    supplied by the state modulation network.
    """

    def __init__(
        self,
        geometry: Any = None,
        hidden_layers: List[int] = None,
        lr: float = 1e-3,
        epochs: int = 20000,
        num_domain: int = 5000,
        num_boundary: int = 500,
        num_initial: int = 200,
        loss_weights: Dict[str, float] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        from .layered_diffusion import SkinLayerGeometry
        self.geom = geometry or SkinLayerGeometry()
        self.lr = lr
        self.epochs = epochs
        self.num_domain = num_domain
        self.num_boundary = num_boundary
        self.num_initial = num_initial
        self.weights = loss_weights or {"pde": 1.0, "bc": 10.0, "ic": 10.0, "data": 50.0}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = SkinPINNNetwork(hidden_layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )

    def _sample_domain(self, n: int, t_max: float) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.rand(n, 1) * self.geom.total_thickness
        t = torch.rand(n, 1) * t_max
        return x.to(self.device), t.to(self.device)

    def _get_D_k(
        self, x: torch.Tensor, params: Dict[str, float]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate spatially-varying D(x) and k(x) from transport parameters."""
        D = torch.where(
            x < self.geom.sc_thickness,
            torch.tensor(params.get("D_sc", 1e-4), device=self.device),
            torch.where(
                x < self.geom.sc_thickness + self.geom.ve_thickness,
                torch.tensor(params.get("D_ve", 1e-2), device=self.device),
                torch.tensor(params.get("D_ve", 1e-2) * 5, device=self.device),
            ),
        )
        k = torch.where(
            x > self.geom.sc_thickness + self.geom.ve_thickness,
            torch.tensor(params.get("k_bind_dermis", 0.0), device=self.device),
            torch.tensor(0.0, device=self.device),
        )
        vasc_start = self.geom.sc_thickness + self.geom.ve_thickness + 0.9 * self.geom.dermis_thickness
        k = k + torch.where(
            x > vasc_start,
            torch.tensor(params.get("k_clear_vasc", 0.01), device=self.device),
            torch.tensor(0.0, device=self.device),
        )
        return D, k

    def _pde_residual(
        self, x: torch.Tensor, t: torch.Tensor, params: Dict[str, float]
    ) -> torch.Tensor:
        """Compute ∂C/∂t − D·∂²C/∂x² + k·C."""
        x.requires_grad_(True)
        t.requires_grad_(True)
        C = self.net(x, t)

        dC_dt = torch.autograd.grad(C, t, torch.ones_like(C), create_graph=True)[0]
        dC_dx = torch.autograd.grad(C, x, torch.ones_like(C), create_graph=True)[0]
        d2C_dx2 = torch.autograd.grad(dC_dx, x, torch.ones_like(dC_dx), create_graph=True)[0]

        D, k = self._get_D_k(x, params)
        residual = dC_dt - D * d2C_dx2 + k * C
        return residual

    def train(
        self,
        params: Dict[str, float],
        t_max_s: float = 48 * 3600,
        C_donor: float = 1000.0,
        observed_data: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, List[float]]:
        """Train the PINN with PDE + BC + IC + data losses.

        Args:
            params: transport parameters (from state modulation or manual)
            t_max_s: total simulation time in seconds
            C_donor: donor concentration
            observed_data: optional dict with 'time_s', 'x', 'concentration'
        """
        history = {"total": [], "pde": [], "bc": [], "ic": [], "data": []}

        for epoch in tqdm(range(self.epochs), desc="PINN training"):
            self.optimizer.zero_grad()

            # Domain points
            x_d, t_d = self._sample_domain(self.num_domain, t_max_s)
            res = self._pde_residual(x_d, t_d, params)
            loss_pde = (res ** 2).mean()

            # BC: surface (x=0) → C = C_donor
            t_bc = torch.rand(self.num_boundary, 1, device=self.device) * t_max_s
            x_bc_top = torch.zeros(self.num_boundary, 1, device=self.device)
            C_top = self.net(x_bc_top, t_bc)
            loss_bc_top = ((C_top - C_donor) ** 2).mean()

            # BC: bottom (x=L) → C = 0 (perfect sink)
            x_bc_bot = torch.full((self.num_boundary, 1), self.geom.total_thickness, device=self.device)
            C_bot = self.net(x_bc_bot, t_bc)
            loss_bc_bot = (C_bot ** 2).mean()

            loss_bc = loss_bc_top + loss_bc_bot

            # IC: t=0 → C = 0 everywhere except surface
            x_ic = torch.rand(self.num_initial, 1, device=self.device) * self.geom.total_thickness
            t_ic = torch.zeros(self.num_initial, 1, device=self.device)
            C_ic = self.net(x_ic, t_ic)
            # At x>0, C should be 0; at x≈0, C = C_donor
            target_ic = torch.where(x_ic < 1.0, C_donor * torch.ones_like(C_ic), torch.zeros_like(C_ic))
            loss_ic = ((C_ic - target_ic) ** 2).mean()

            # Data loss
            loss_data = torch.tensor(0.0, device=self.device)
            if observed_data is not None:
                x_obs = torch.tensor(observed_data["x"], dtype=torch.float32, device=self.device).unsqueeze(-1)
                t_obs = torch.tensor(observed_data["time_s"], dtype=torch.float32, device=self.device).unsqueeze(-1)
                C_obs = torch.tensor(observed_data["concentration"], dtype=torch.float32, device=self.device).unsqueeze(-1)
                C_pred = self.net(x_obs, t_obs)
                loss_data = ((C_pred - C_obs) ** 2).mean()

            loss = (
                self.weights["pde"] * loss_pde
                + self.weights["bc"] * loss_bc
                + self.weights["ic"] * loss_ic
                + self.weights["data"] * loss_data
            )

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            history["total"].append(loss.item())
            history["pde"].append(loss_pde.item())
            history["bc"].append(loss_bc.item())
            history["ic"].append(loss_ic.item())
            history["data"].append(loss_data.item())

        return history

    def predict(
        self, x: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        """Predict C(x, t) on arbitrary (x, t) grid."""
        self.net.eval()
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(-1)
        t_t = torch.tensor(t, dtype=torch.float32, device=self.device).unsqueeze(-1)
        with torch.no_grad():
            C = self.net(x_t, t_t)
        return C.cpu().numpy().flatten()

    def predict_cumulative_permeation(
        self, t_eval_s: np.ndarray, params: Dict[str, float]
    ) -> np.ndarray:
        """Integrate flux at dermis bottom over time → cumulative permeation."""
        x_bot = self.geom.total_thickness
        eps = 1.0  # µm offset for gradient
        C_bot = self.predict(np.full_like(t_eval_s, x_bot), t_eval_s)
        C_near = self.predict(np.full_like(t_eval_s, x_bot - eps), t_eval_s)

        D_dermis = params.get("D_ve", 1e-2) * 5
        flux = D_dermis * (C_near - C_bot) / eps  # µg/µm²/s
        cum = np.cumsum(flux * np.gradient(t_eval_s)) * 1e8  # → µg/cm²
        return cum
