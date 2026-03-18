"""
Blind extrapolation testing.

Test whether the virtual skin truly learns state-dependent transport rules
rather than memorising training data.

Holdout categories:
  - Unseen donor
  - Unseen anatomical site
  - Unseen skin condition / pathology
  - Unseen drug
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .metrics import ValidationMetrics
from ..transport.drug_transport import DrugTransportPredictor, TransportPrediction
from ..atlas.state_space import TissueStateVector
from ..data.drug_properties import DrugDescriptor
from ..data.skin_sample import SkinSample


class HoldoutType(Enum):
    UNSEEN_DONOR = "unseen_donor"
    UNSEEN_SITE = "unseen_site"
    UNSEEN_CONDITION = "unseen_condition"
    UNSEEN_DRUG = "unseen_drug"


class BlindExtrapolationTest:
    """Run blind tests across different holdout categories."""

    def __init__(self, predictor: DrugTransportPredictor) -> None:
        self.predictor = predictor

    def run_blind_test(
        self,
        test_samples: List[Dict[str, Any]],
        holdout_type: HoldoutType,
    ) -> pd.DataFrame:
        """Execute blind predictions and compare to held-out observations.

        Each entry in test_samples should contain:
          - 'tissue_state': TissueStateVector
          - 'drug': DrugDescriptor
          - 'observed_Jss': float
          - 'observed_lag_time': float
          - 'observed_Q_total': float
          - optional: 'observed_sc_retention', 'observed_dermis_retention'
          - 'sample_id': str
        """
        rows = []
        for sample in test_samples:
            ts = sample["tissue_state"]
            drug = sample["drug"]
            pred = self.predictor.predict(ts, drug)

            row = {
                "sample_id": sample.get("sample_id", "unknown"),
                "holdout_type": holdout_type.value,
                "pred_Jss": pred.steady_state_flux,
                "pred_lag_time": pred.lag_time,
                "pred_Q_total": float(pred.cumulative_permeation[-1]),
                "pred_target_AUC": pred.target_layer_auc,
            }

            for key in ["observed_Jss", "observed_lag_time", "observed_Q_total"]:
                if key in sample:
                    row[key] = sample[key]

            rows.append(row)

        return pd.DataFrame(rows)

    def evaluate_extrapolation(
        self, results_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Compute metrics on blind test results."""
        metrics = {}

        for endpoint in ["Jss", "lag_time", "Q_total"]:
            obs_col = f"observed_{endpoint}"
            pred_col = f"pred_{endpoint}"
            if obs_col in results_df and pred_col in results_df:
                obs = results_df[obs_col].dropna().values
                pred = results_df[pred_col].dropna().values
                n = min(len(obs), len(pred))
                if n >= 2:
                    report = ValidationMetrics.full_report(obs[:n], pred[:n])
                    for k, v in report.items():
                        metrics[f"blind_{endpoint}_{k}"] = v

        return metrics

    def uncertainty_degradation(
        self,
        train_metrics: Dict[str, float],
        blind_metrics: Dict[str, float],
    ) -> Dict[str, float]:
        """Check whether uncertainty appropriately increases on OOD samples."""
        degradation = {}
        for key in train_metrics:
            blind_key = f"blind_{key.split('_', 1)[-1]}" if not key.startswith("blind_") else key
            if blind_key in blind_metrics:
                degradation[key] = blind_metrics[blind_key] - train_metrics[key]
        return degradation

    def failure_mode_analysis(
        self, results_df: pd.DataFrame, fold_threshold: float = 3.0
    ) -> pd.DataFrame:
        """Identify samples where predictions fail beyond acceptable fold error."""
        failures = []
        for _, row in results_df.iterrows():
            for endpoint in ["Jss", "lag_time", "Q_total"]:
                obs = row.get(f"observed_{endpoint}")
                pred = row.get(f"pred_{endpoint}")
                if obs is not None and pred is not None and obs > 0:
                    fe = max(pred / obs, obs / pred)
                    if fe > fold_threshold:
                        failures.append({
                            "sample_id": row.get("sample_id"),
                            "endpoint": endpoint,
                            "fold_error": fe,
                            "observed": obs,
                            "predicted": pred,
                        })
        return pd.DataFrame(failures)
