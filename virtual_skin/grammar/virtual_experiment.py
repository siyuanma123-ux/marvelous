"""
Virtual perturbation experiments — in silico "thought experiments".

Enables:
  1. What-if simulations: perturb a state axis and predict endpoint changes
  2. Counterfactual analysis: disable a mechanism module and observe effect
  3. Experimental design suggestions: which perturbation is most informative?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..atlas.state_space import TissueStateVector
from ..transport.drug_transport import DrugTransportPredictor, TransportPrediction
from ..data.drug_properties import DrugDescriptor
from .rule_engine import RuleEngine


@dataclass
class VirtualExperiment:
    """Definition and results of a virtual perturbation experiment."""

    experiment_id: str
    description: str

    # Perturbation specification
    perturbed_axis: Optional[str] = None
    perturbation_type: str = "set"  # "set", "increase", "decrease", "knockout"
    perturbation_value: float = 0.0

    # Or rule-level perturbation
    disabled_rules: List[str] = field(default_factory=list)

    # Results
    baseline_prediction: Optional[TransportPrediction] = None
    perturbed_prediction: Optional[TransportPrediction] = None

    # Derived metrics
    delta_flux: float = 0.0
    delta_lag_time: float = 0.0
    delta_auc: float = 0.0

    def compute_deltas(self) -> None:
        if self.baseline_prediction and self.perturbed_prediction:
            self.delta_flux = (
                self.perturbed_prediction.steady_state_flux
                - self.baseline_prediction.steady_state_flux
            )
            self.delta_lag_time = (
                self.perturbed_prediction.lag_time
                - self.baseline_prediction.lag_time
            )
            self.delta_auc = (
                self.perturbed_prediction.target_layer_auc
                - self.baseline_prediction.target_layer_auc
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "description": self.description,
            "perturbed_axis": self.perturbed_axis,
            "perturbation_value": self.perturbation_value,
            "delta_flux": self.delta_flux,
            "delta_lag_time": self.delta_lag_time,
            "delta_auc": self.delta_auc,
            "baseline_Jss": self.baseline_prediction.steady_state_flux if self.baseline_prediction else None,
            "perturbed_Jss": self.perturbed_prediction.steady_state_flux if self.perturbed_prediction else None,
        }


class CounterfactualSimulator:
    """Run virtual experiments on the virtual skin system."""

    def __init__(
        self,
        predictor: DrugTransportPredictor,
        rule_engine: Optional[RuleEngine] = None,
    ) -> None:
        self.predictor = predictor
        self.rule_engine = rule_engine or RuleEngine()

    def perturb_state_axis(
        self,
        tissue_state: TissueStateVector,
        drug: DrugDescriptor,
        axis: str,
        target_value: float,
        experiment_id: str = "VE001",
    ) -> VirtualExperiment:
        """Set one state axis to a target value and compare predictions."""
        # Baseline
        baseline = self.predictor.predict(tissue_state, drug)

        # Perturbed state
        arr = tissue_state.to_array().copy()
        idx = TissueStateVector.axis_names().index(axis)
        arr[idx] = target_value
        perturbed_ts = TissueStateVector(
            barrier_integrity=arr[0],
            inflammatory_load=arr[1],
            ecm_remodeling=arr[2],
            vascularization=arr[3],
            appendage_openness=arr[4],
        )
        perturbed = self.predictor.predict(perturbed_ts, drug)

        exp = VirtualExperiment(
            experiment_id=experiment_id,
            description=f"Set {axis} from {tissue_state.to_array()[idx]:.2f} to {target_value:.2f}",
            perturbed_axis=axis,
            perturbation_type="set",
            perturbation_value=target_value,
            baseline_prediction=baseline,
            perturbed_prediction=perturbed,
        )
        exp.compute_deltas()
        return exp

    def knockout_module(
        self,
        tissue_state: TissueStateVector,
        drug: DrugDescriptor,
        rule_ids: List[str],
        experiment_id: str = "VE_KO",
    ) -> VirtualExperiment:
        """Disable specific grammar rules and predict the effect."""
        base_params = self.predictor.mod_net.predict_params(
            tissue_state, drug.to_vector()
        )

        # Baseline with all rules
        params_full, _ = self.rule_engine.evaluate(tissue_state, base_params)
        # Knockout
        params_ko = self.rule_engine.counterfactual(
            tissue_state, base_params, disable_rules=rule_ids
        )

        # Predict both
        baseline = self.predictor.predict(tissue_state, drug)

        # For knockout: temporarily swap modulation
        original_net = self.predictor.mod_net
        self.predictor.mod_net = _FixedParamNet(params_ko)
        perturbed = self.predictor.predict(tissue_state, drug)
        self.predictor.mod_net = original_net

        exp = VirtualExperiment(
            experiment_id=experiment_id,
            description=f"Knockout rules: {rule_ids}",
            disabled_rules=rule_ids,
            baseline_prediction=baseline,
            perturbed_prediction=perturbed,
        )
        exp.compute_deltas()
        return exp

    def full_sweep(
        self,
        tissue_state: TissueStateVector,
        drug: DrugDescriptor,
        n_points: int = 10,
    ) -> pd.DataFrame:
        """Sweep all state axes and compile results."""
        rows = []
        for axis in TissueStateVector.axis_names():
            for val in np.linspace(0, 1, n_points):
                exp = self.perturb_state_axis(
                    tissue_state, drug, axis, val,
                    experiment_id=f"sweep_{axis}_{val:.2f}",
                )
                row = exp.to_dict()
                rows.append(row)
        return pd.DataFrame(rows)

    def suggest_most_informative_experiment(
        self,
        tissue_state: TissueStateVector,
        drug: DrugDescriptor,
    ) -> Dict[str, Any]:
        """Identify which perturbation produces the largest predicted effect."""
        max_delta = 0.0
        best = None
        for axis in TissueStateVector.axis_names():
            for val in [0.0, 1.0]:
                exp = self.perturb_state_axis(tissue_state, drug, axis, val)
                total_delta = abs(exp.delta_flux) + abs(exp.delta_auc)
                if total_delta > max_delta:
                    max_delta = total_delta
                    best = exp
        if best:
            return {
                "recommended_perturbation": best.description,
                "expected_delta_flux": best.delta_flux,
                "expected_delta_auc": best.delta_auc,
            }
        return {"recommended_perturbation": "none"}


class _FixedParamNet:
    """Mock modulation network that returns fixed parameters."""

    def __init__(self, params: Dict[str, float]) -> None:
        self._params = params

    def predict_params(self, tissue_state, drug_vector):
        return dict(self._params)
