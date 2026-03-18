"""
Perturbation experiment validation.

Three classes of directed perturbation:
  1. Inflammatory myofibroblast / ECM remodeling module
  2. Macrophage–endothelial module
  3. Keratinocyte barrier state
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .metrics import ValidationMetrics


@dataclass
class PerturbationResult:
    """Result of a single perturbation experiment."""

    experiment_id: str
    perturbation_type: str
    module_name: str

    # Observed changes
    observed_endpoint_changes: Dict[str, float] = field(default_factory=dict)

    # Model-predicted changes
    predicted_endpoint_changes: Dict[str, float] = field(default_factory=dict)

    # Omics markers measured
    observed_markers: Dict[str, float] = field(default_factory=dict)

    # Direction concordance
    direction_concordance: Optional[bool] = None

    def compute_concordance(self) -> bool:
        """Check whether predicted and observed changes agree in direction."""
        concordant = True
        for key in self.observed_endpoint_changes:
            if key in self.predicted_endpoint_changes:
                obs_dir = np.sign(self.observed_endpoint_changes[key])
                pred_dir = np.sign(self.predicted_endpoint_changes[key])
                if obs_dir != pred_dir and obs_dir != 0:
                    concordant = False
        self.direction_concordance = concordant
        return concordant


class PerturbationValidator:
    """Validate model mechanism predictions against perturbation experiments."""

    def __init__(self) -> None:
        self.results: List[PerturbationResult] = []

    def add_result(self, result: PerturbationResult) -> None:
        result.compute_concordance()
        self.results.append(result)

    def direction_concordance_rate(self) -> float:
        """Fraction of perturbations where model direction matches experiment."""
        if not self.results:
            return 0.0
        return float(np.mean([r.direction_concordance for r in self.results if r.direction_concordance is not None]))

    def magnitude_correlation(self, endpoint: str = "Jss") -> float:
        """Correlation between predicted and observed magnitude changes."""
        obs, pred = [], []
        for r in self.results:
            if endpoint in r.observed_endpoint_changes and endpoint in r.predicted_endpoint_changes:
                obs.append(r.observed_endpoint_changes[endpoint])
                pred.append(r.predicted_endpoint_changes[endpoint])
        if len(obs) < 3:
            return float("nan")
        return float(np.corrcoef(obs, pred)[0, 1])

    def summary(self) -> pd.DataFrame:
        rows = []
        for r in self.results:
            row = {
                "experiment_id": r.experiment_id,
                "perturbation_type": r.perturbation_type,
                "module": r.module_name,
                "direction_concordant": r.direction_concordance,
            }
            for k, v in r.observed_endpoint_changes.items():
                row[f"obs_{k}"] = v
            for k, v in r.predicted_endpoint_changes.items():
                row[f"pred_{k}"] = v
            rows.append(row)
        return pd.DataFrame(rows)

    def generate_perturbation_plan(self) -> List[Dict[str, str]]:
        """Suggest experiments based on model-predicted high-impact perturbations."""
        return [
            {
                "module": "inflammatory_myofibroblast",
                "perturbation": "Induce inflammatory fibroblast state via TGFβ + IL-1β treatment",
                "readout": "ECM markers, dermal retention, target-layer AUC",
                "platform": "Augmented HSE or ex vivo skin",
            },
            {
                "module": "macrophage_endothelial",
                "perturbation": "Add/remove macrophages in vascularised HSE",
                "readout": "Vascular clearance proxy, exposure redistribution",
                "platform": "Immune-augmented HSE or skin chip",
            },
            {
                "module": "keratinocyte_barrier",
                "perturbation": "Modulate EMT-like programme via ZEB1 overexpression or TGFβ",
                "readout": "TEER, Jss, lag time, barrier marker IF",
                "platform": "Augmented HSE with controlled O₂",
            },
        ]
