"""
State-dependent parameter modulation network.

Maps low-dimensional tissue state axes + drug descriptors → transport parameters.

Design philosophy (from the proposal):
  "The state-modulation layer does NOT receive high-dimensional gene expression
   directly, but only barrier_integrity, inflammatory_load, ecm_remodeling,
   vascularization, appendage_openness, plus drug MW, logP, pKa, charge,
   binding_affinity."
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..atlas.state_space import TissueStateVector, STATE_AXIS_NAMES


# Default bounds for each open transport parameter (log10 scale, µm²/s for D)
# PDE solver uses µm²/s internally; 1 cm²/s = 10^8 µm²/s
PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "D_sc": (-4.0, 0.0),           # log10(µm²/s), typ 0.001–1
    "K_sc_ve": (-0.5, 2.0),        # log10(partition), typ 0.3–100
    "D_ve": (-1.0, 3.0),           # log10(µm²/s), typ 0.1–1000
    "k_bind_dermis": (-7.0, -3.0),  # log10(1/s)
    "k_clear_vasc": (-5.0, -2.0),
    "w_appendage": (-6.0, -3.0),
}

OPEN_PARAM_NAMES = ["D_sc", "K_sc_ve", "D_ve", "k_bind_dermis", "k_clear_vasc", "w_appendage"]


class StateModulationNetwork(nn.Module):
    """Lightweight MLP: (state_axes ⊕ drug_descriptors) → open transport parameters.

    Uses sigmoid output scaled to physically plausible bounds for each parameter,
    ensuring identifiability and preventing parameter explosion.
    """

    def __init__(
        self,
        n_state_axes: int = 5,
        n_drug_desc: int = 8,
        hidden_dims: List[int] | None = None,
        param_names: List[str] | None = None,
        param_bounds: Dict[str, Tuple[float, float]] | None = None,
    ) -> None:
        super().__init__()
        self.param_names = param_names or OPEN_PARAM_NAMES
        self.bounds = param_bounds or PARAM_BOUNDS
        n_out = len(self.param_names)
        hidden_dims = hidden_dims or [32, 16]

        layers: List[nn.Module] = []
        in_dim = n_state_axes + n_drug_desc
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, n_out))
        self.net = nn.Sequential(*layers)

        # Register bounds as buffers
        lo = torch.tensor([self.bounds.get(p, (-5, 0))[0] for p in self.param_names])
        hi = torch.tensor([self.bounds.get(p, (-5, 0))[1] for p in self.param_names])
        self.register_buffer("lo", lo)
        self.register_buffer("hi", hi)

    def forward(
        self,
        state_vec: torch.Tensor,
        drug_vec: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            state_vec: (batch, n_state_axes)
            drug_vec:  (batch, n_drug_desc)
        Returns:
            dict mapping parameter name → (batch,) tensor in physical units
        """
        x = torch.cat([state_vec, drug_vec], dim=-1)
        raw = self.net(x)  # (batch, n_params) in (-∞, +∞)
        scaled = torch.sigmoid(raw) * (self.hi - self.lo) + self.lo  # log-scale
        params_phys = 10.0 ** scaled  # convert log10 → linear

        return {name: params_phys[:, i] for i, name in enumerate(self.param_names)}

    def predict_params(
        self,
        tissue_state: TissueStateVector,
        drug_vector: np.ndarray,
    ) -> Dict[str, float]:
        """Convenience: single-sample numpy → dict of floats."""
        sv = torch.tensor(tissue_state.to_array(), dtype=torch.float32).unsqueeze(0)
        dv = torch.tensor(drug_vector, dtype=torch.float32).unsqueeze(0)
        self.eval()
        with torch.no_grad():
            out = self.forward(sv, dv)
        return {k: float(v.item()) for k, v in out.items()}


class ParameterPrior:
    """Encode domain-knowledge priors on each transport parameter.

    Used by the Bayesian calibrator to regularize the modulation network.
    """

    def __init__(self) -> None:
        # (mean_log10, std_log10) for each parameter — in PDE units (µm²/s for D)
        self.priors: Dict[str, Tuple[float, float]] = {
            "D_sc": (-2.5, 1.0),       # typ 0.003 µm²/s
            "K_sc_ve": (0.5, 0.8),      # typ ~3
            "D_ve": (1.5, 1.0),         # typ ~30 µm²/s
            "k_bind_dermis": (-5.0, 1.0),
            "k_clear_vasc": (-3.5, 0.8),
            "w_appendage": (-5.0, 1.0),
        }

    def log_prob(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Sum of Gaussian log-priors on log10-scale."""
        total = torch.tensor(0.0)
        for name, val in params.items():
            if name in self.priors:
                mu, sigma = self.priors[name]
                log_val = torch.log10(val + 1e-30)
                total = total + (-0.5 * ((log_val - mu) / sigma) ** 2).sum()
        return total
