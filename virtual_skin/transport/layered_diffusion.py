"""
Layered Fick's diffusion PDE for multi-layer skin.

Physical backbone:
  ∂C/∂t = D_i · ∂²C/∂x² − k_bind_i · C + S_i    for each layer i
  
  Interface conditions:
    C_i(x_interface) / C_{i+1}(x_interface) = K_{i,i+1}  (partition)
    D_i · ∂C_i/∂x = D_{i+1} · ∂C_{i+1}/∂x              (flux continuity)

  Boundary conditions:
    x = 0 (skin surface):  C = C_donor  (infinite dose) or flux BC (finite dose)
    x = L (dermis bottom):  C = 0       (perfect sink / vascular clearance)

Open state-dependent parameters:
  D_sc, K_sc_ve, D_ve, k_bind_dermis, k_clear_vasc, [w_appendage]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_banded


@dataclass
class SkinLayerGeometry:
    """Physical geometry of the skin system."""

    sc_thickness: float = 15.0       # µm
    ve_thickness: float = 80.0
    dermis_thickness: float = 300.0  # dermatomed skin for IVPT (~400µm total)
    appendage_frac: float = 0.001    # fractional area of follicular route

    n_sc: int = 30
    n_ve: int = 15
    n_dermis: int = 30

    @property
    def total_thickness(self) -> float:
        return self.sc_thickness + self.ve_thickness + self.dermis_thickness

    @property
    def layer_boundaries(self) -> np.ndarray:
        """Cumulative boundaries [0, sc, sc+ve, sc+ve+dermis] in µm."""
        return np.array([
            0.0,
            self.sc_thickness,
            self.sc_thickness + self.ve_thickness,
            self.total_thickness,
        ])

    def build_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (x_nodes, layer_ids) over the full skin thickness."""
        x_sc = np.linspace(0, self.sc_thickness, self.n_sc, endpoint=False)
        x_ve = np.linspace(
            self.sc_thickness, self.sc_thickness + self.ve_thickness, self.n_ve, endpoint=False
        )
        x_de = np.linspace(
            self.sc_thickness + self.ve_thickness, self.total_thickness, self.n_dermis
        )
        x = np.concatenate([x_sc, x_ve, x_de])
        layers = np.concatenate([
            np.zeros(self.n_sc, dtype=int),
            np.ones(self.n_ve, dtype=int),
            2 * np.ones(self.n_dermis, dtype=int),
        ])
        return x, layers


@dataclass
class TransportParameters:
    """Physical parameters for the layered diffusion model."""

    D_sc: float = 1e-4          # cm²/s → actually µm²/s in model units
    D_ve: float = 1e-2
    D_dermis: float = 5e-2
    K_sc_ve: float = 10.0       # SC/VE partition coefficient
    K_ve_dermis: float = 1.0    # VE/dermis partition coefficient
    k_bind_dermis: float = 0.0  # first-order dermal binding rate (1/s)
    k_clear_vasc: float = 0.01  # vascular clearance rate at dermis bottom (1/s)
    w_appendage: float = 0.0    # appendage bypass weight

    C_donor: float = 1000.0     # donor concentration (µg/cm³)

    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items()}


class LayeredDiffusionPDE:
    """Finite-difference solver for the state-dependent layered skin diffusion PDE.

    Uses method-of-lines: spatial discretization → ODE system → time integration.
    """

    def __init__(
        self,
        geometry: Optional[SkinLayerGeometry] = None,
        params: Optional[TransportParameters] = None,
    ) -> None:
        self.geom = geometry or SkinLayerGeometry()
        self.params = params or TransportParameters()

    def _build_diffusivity_field(self, x: np.ndarray, layers: np.ndarray) -> np.ndarray:
        D = np.zeros_like(x)
        D[layers == 0] = self.params.D_sc
        D[layers == 1] = self.params.D_ve
        D[layers == 2] = self.params.D_dermis
        return D

    def _build_sink_field(self, x: np.ndarray, layers: np.ndarray) -> np.ndarray:
        """Local sink/binding rate k(x)."""
        k = np.zeros_like(x)
        k[layers == 2] = self.params.k_bind_dermis
        # Enhanced clearance at the dermal–vascular interface (last 10% of dermis)
        dermis_end = self.geom.total_thickness
        dermis_start = self.geom.sc_thickness + self.geom.ve_thickness
        vasc_zone = dermis_start + 0.9 * (dermis_end - dermis_start)
        k[x > vasc_zone] += self.params.k_clear_vasc
        return k

    def solve(
        self,
        t_total_s: float = 48 * 3600,
        dt_output_s: float = 1800,
        dose_type: str = "infinite",
    ) -> Dict[str, np.ndarray]:
        """Solve the diffusion PDE and return pharmacokinetic outputs.

        Returns dict with keys:
          time_h, concentration_profile, cumulative_permeation,
          flux_profile, layer_retention
        """
        x, layers = self.geom.build_grid()
        N = len(x)
        dx = np.diff(x)

        D = self._build_diffusivity_field(x, layers)
        k = self._build_sink_field(x, layers)

        # Handle partition at interfaces: scale concentration jump
        # Interface at SC/VE
        isc = self.geom.n_sc - 1
        ive = self.geom.n_sc
        # Interface at VE/dermis
        ive_end = self.geom.n_sc + self.geom.n_ve - 1
        ide = self.geom.n_sc + self.geom.n_ve

        t_eval = np.arange(0, t_total_s + dt_output_s, dt_output_s)

        K_sc_ve = max(self.params.K_sc_ve, 1e-6)
        K_ve_de = max(self.params.K_ve_dermis, 1e-6)

        def rhs(t, C):
            dCdt = np.zeros(N)
            for i in range(1, N - 1):
                dx_l = x[i] - x[i - 1]
                dx_r = x[i + 1] - x[i]
                dx_avg = 0.5 * (dx_l + dx_r)

                D_l = 2 * D[i - 1] * D[i] / (D[i - 1] + D[i] + 1e-30)
                D_r = 2 * D[i] * D[i + 1] / (D[i] + D[i + 1] + 1e-30)

                C_left = C[i - 1]
                C_right = C[i + 1]

                # Partition corrections at BOTH sides of each interface
                # SC→VE: C_sc = K_sc_ve × C_ve at equilibrium
                if i == isc:
                    # Last SC node: right neighbor is VE; convert C_ve to SC-equivalent
                    C_right = K_sc_ve * C[i + 1]
                elif i == ive:
                    # First VE node: left neighbor is SC; convert C_sc to VE-equivalent
                    C_left = C[i - 1] / K_sc_ve
                # VE→dermis: C_ve = K_ve_de × C_dermis
                elif i == ive_end:
                    C_right = K_ve_de * C[i + 1]
                elif i == ide:
                    C_left = C[i - 1] / K_ve_de

                flux_l = D_l * (C_left - C[i]) / dx_l
                flux_r = D_r * (C_right - C[i]) / dx_r

                dCdt[i] = np.clip((flux_l + flux_r) / dx_avg - k[i] * C[i], -1e10, 1e10)

            # Boundary conditions
            if dose_type == "infinite":
                dCdt[0] = 0  # C[0] = C_donor (held fixed below)
            else:
                # Finite dose: no-flux at surface after depletion
                dCdt[0] = D[0] * (C[1] - C[0]) / (dx[0] ** 2) - k[0] * C[0]

            # Perfect sink at bottom
            dCdt[-1] = 0  # C[-1] = 0

            return dCdt

        # Initial condition
        C0 = np.zeros(N)
        if dose_type == "infinite":
            C0[0] = self.params.C_donor

        sol = solve_ivp(
            rhs, [0, t_total_s], C0, t_eval=t_eval, method="Radau",
            rtol=1e-4, atol=1e-7, max_step=3600,
        )

        if not sol.success:
            raise RuntimeError(f"PDE solver failed: {sol.message}")

        C_profiles = sol.y.T  # (n_times, n_nodes)
        times_h = sol.t / 3600.0

        # Enforce BCs in output
        if dose_type == "infinite":
            C_profiles[:, 0] = self.params.C_donor
        C_profiles[:, -1] = 0.0

        # Compute flux at receptor (bottom of dermis)
        # D in µm²/s, C in µg/cm³, dx in µm → flux in µg/(cm³) * (µm/s) = µg·µm/(cm³·s)
        flux_raw = D[-1] * (C_profiles[:, -2] - C_profiles[:, -1]) / dx[-1]

        # Convert to µg/(cm²·s): multiply by 1e-4 (since 1 µm = 1e-4 cm)
        flux = flux_raw * 1e-4

        # Cumulative permeation: ∫flux dt → µg/cm²
        cum_perm_cm2 = np.cumsum(flux * np.gradient(sol.t))

        # Layer retention at final time
        C_final = C_profiles[-1]
        retention = {
            "sc": float(np.trapezoid(C_final[layers == 0], x[layers == 0])),
            "ve": float(np.trapezoid(C_final[layers == 1], x[layers == 1])),
            "dermis": float(np.trapezoid(C_final[layers == 2], x[layers == 2])),
        }

        # Appendage bypass contribution (simplified parallel pathway)
        if self.params.w_appendage > 0:
            app_flux = self.params.w_appendage * self.params.C_donor * self.params.D_ve / self.geom.ve_thickness
            cum_perm_cm2 += app_flux * sol.t * 1e-4

        # Flux in µg/(cm²·h) for convenience
        flux_per_h = flux * 3600.0

        return {
            "time_h": times_h,
            "concentration_profile": C_profiles,
            "x_grid": x,
            "layer_ids": layers,
            "cumulative_permeation": cum_perm_cm2,
            "flux": flux_per_h,
            "layer_retention": retention,
        }

    def steady_state_flux(self, result: Dict[str, np.ndarray]) -> float:
        t, q = result["time_h"], result["cumulative_permeation"]
        half = len(t) // 2
        if half < 2:
            return 0.0
        coeffs = np.polyfit(t[half:], q[half:], 1)
        return float(coeffs[0])

    def lag_time(self, result: Dict[str, np.ndarray]) -> float:
        jss = self.steady_state_flux(result)
        if jss <= 0:
            return float("inf")
        t, q = result["time_h"], result["cumulative_permeation"]
        half = len(t) // 2
        coeffs = np.polyfit(t[half:], q[half:], 1)
        return float(-coeffs[1] / coeffs[0])
