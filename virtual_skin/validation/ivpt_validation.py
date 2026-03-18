"""
IVPT / Franz diffusion cell validation framework.

Primary pharmacological truth source for the virtual skin system.
Supports multi-drug, multi-formulation, multi-site, multi-state designs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.ivpt_data import FranzDiffusionData, IVPTExperiment
from ..transport.drug_transport import DrugTransportPredictor, TransportPrediction
from ..atlas.state_space import TissueStateVector
from ..data.drug_properties import DrugDescriptor
from .metrics import ValidationMetrics


class IVPTValidator:
    """Validate virtual skin predictions against IVPT experimental data."""

    def __init__(self, predictor: DrugTransportPredictor) -> None:
        self.predictor = predictor
        self.results: List[Dict[str, Any]] = []

    def validate_single(
        self,
        observed: FranzDiffusionData,
        tissue_state: TissueStateVector,
        drug: DrugDescriptor,
    ) -> Dict[str, Any]:
        """Compare prediction vs. single Franz cell run."""
        pred = self.predictor.predict(tissue_state, drug)
        obs = observed

        # Align time grids
        t_common = np.intersect1d(
            np.round(pred.time_h, 2), np.round(obs.time_h, 2)
        )
        if len(t_common) < 3:
            t_common = pred.time_h

        pred_q = np.interp(t_common, pred.time_h, pred.cumulative_permeation)
        obs_q = np.interp(t_common, obs.time_h, obs.cumulative_permeation)

        result = {
            "sample_id": obs.sample_id,
            "drug": obs.drug_name,
            "obs_Jss": obs.steady_state_flux(),
            "pred_Jss": pred.steady_state_flux,
            "obs_lag_time": obs.lag_time(),
            "pred_lag_time": pred.lag_time,
            "obs_Q_total": float(obs.cumulative_permeation[-1]),
            "pred_Q_total": float(pred.cumulative_permeation[-1]),
            "curve_RMSE": ValidationMetrics.rmse(obs_q, pred_q),
            "curve_R2": ValidationMetrics.r2(obs_q, pred_q),
        }

        # Layer retention comparison
        if obs.sc_retention is not None and "sc" in pred.layer_retention:
            result["obs_sc_retention"] = obs.sc_retention
            result["pred_sc_retention"] = pred.layer_retention["sc"]
        if obs.dermis_retention is not None and "dermis" in pred.layer_retention:
            result["obs_dermis_retention"] = obs.dermis_retention
            result["pred_dermis_retention"] = pred.layer_retention["dermis"]

        self.results.append(result)
        return result

    def validate_experiment(
        self,
        experiment: IVPTExperiment,
        tissue_states: List[TissueStateVector],
        drug: DrugDescriptor,
    ) -> pd.DataFrame:
        """Validate across all runs in an IVPT experiment."""
        for run, ts in zip(experiment.runs, tissue_states):
            self.validate_single(run, ts, drug)
        return self.summary()

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)

    def aggregate_metrics(self) -> Dict[str, float]:
        """Compute aggregate metrics across all validated samples."""
        df = self.summary()
        if df.empty:
            return {}

        metrics = {}
        for endpoint in ["Jss", "lag_time", "Q_total"]:
            obs_col = f"obs_{endpoint}"
            pred_col = f"pred_{endpoint}"
            if obs_col in df and pred_col in df:
                obs = df[obs_col].dropna().values
                pred = df[pred_col].dropna().values
                n = min(len(obs), len(pred))
                if n > 0:
                    report = ValidationMetrics.full_report(obs[:n], pred[:n])
                    for k, v in report.items():
                        metrics[f"{endpoint}_{k}"] = v
        return metrics
