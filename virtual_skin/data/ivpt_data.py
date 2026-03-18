"""
In Vitro Permeation Test (IVPT) / Franz diffusion cell data handling.

Endpoints captured:
  - cumulative_permeation(t)  [µg/cm²]
  - steady-state flux Jss     [µg/cm²/h]
  - lag time tlag              [h]
  - layer-specific retention   [µg/cm²] per layer
  - target-layer AUC / Cmax
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FranzDiffusionData:
    """Raw time-course data from a single Franz cell run."""

    sample_id: str
    drug_name: str
    formulation: str

    time_h: np.ndarray                   # sampling time points (h)
    cumulative_permeation: np.ndarray    # µg/cm²
    receptor_concentration: Optional[np.ndarray] = None  # µg/mL

    # Layer retention at experiment end (tape-strip + tissue extraction)
    sc_retention: Optional[float] = None        # µg/cm²
    ve_retention: Optional[float] = None
    dermis_retention: Optional[float] = None

    diffusion_area_cm2: float = 1.77     # typical Franz cell area
    receptor_volume_mL: float = 12.0

    metadata: Dict = field(default_factory=dict)

    # ---- Derived pharmacokinetic endpoints ----

    def steady_state_flux(self, window: Tuple[float, float] | None = None) -> float:
        """Jss from linear portion of cumulative curve (µg/cm²/h)."""
        t, q = self.time_h, self.cumulative_permeation
        if window is not None:
            mask = (t >= window[0]) & (t <= window[1])
            t, q = t[mask], q[mask]
        else:
            half = len(t) // 2
            t, q = t[half:], q[half:]
        if len(t) < 2:
            return 0.0
        coeffs = np.polyfit(t, q, 1)
        return float(coeffs[0])

    def lag_time(self) -> float:
        """Lag time from x-intercept of the steady-state linear fit (h)."""
        jss = self.steady_state_flux()
        if jss <= 0:
            return float("inf")
        t, q = self.time_h, self.cumulative_permeation
        half = len(t) // 2
        t_ss, q_ss = t[half:], q[half:]
        coeffs = np.polyfit(t_ss, q_ss, 1)
        return float(-coeffs[1] / coeffs[0])

    def total_retention(self) -> float:
        vals = [
            v
            for v in (self.sc_retention, self.ve_retention, self.dermis_retention)
            if v is not None
        ]
        return sum(vals)

    def to_dict(self) -> Dict:
        return {
            "sample_id": self.sample_id,
            "drug": self.drug_name,
            "formulation": self.formulation,
            "Jss": self.steady_state_flux(),
            "lag_time": self.lag_time(),
            "Q_total": float(self.cumulative_permeation[-1]),
            "sc_retention": self.sc_retention,
            "ve_retention": self.ve_retention,
            "dermis_retention": self.dermis_retention,
        }


@dataclass
class IVPTExperiment:
    """Collection of Franz runs sharing the same drug × formulation design."""

    drug_name: str
    formulation: str
    runs: List[FranzDiffusionData] = field(default_factory=list)

    def add_run(self, run: FranzDiffusionData) -> None:
        self.runs.append(run)

    def summary_table(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.runs])

    def mean_flux_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (time, mean_Q, std_Q) aligned to common time grid."""
        all_t = np.unique(np.concatenate([r.time_h for r in self.runs]))
        interped = []
        for r in self.runs:
            interped.append(np.interp(all_t, r.time_h, r.cumulative_permeation))
        mat = np.stack(interped, axis=0)
        return all_t, mat.mean(axis=0), mat.std(axis=0)

    def split_train_test(
        self, test_fraction: float = 0.2, seed: int = 42
    ) -> Tuple["IVPTExperiment", "IVPTExperiment"]:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(self.runs))
        n_test = max(1, int(len(self.runs) * test_fraction))
        test_idx, train_idx = idx[:n_test], idx[n_test:]
        train_exp = IVPTExperiment(self.drug_name, self.formulation)
        test_exp = IVPTExperiment(self.drug_name, self.formulation)
        for i in train_idx:
            train_exp.add_run(self.runs[i])
        for i in test_idx:
            test_exp.add_run(self.runs[i])
        return train_exp, test_exp
