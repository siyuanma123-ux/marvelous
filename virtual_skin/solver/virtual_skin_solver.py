"""
Integrated Virtual Skin Solver.

Orchestrates the full pipeline:
  omics data → state space → communication modules → grammar rules
  → state-modulated transport parameters → PDE solve → pharmacokinetic endpoints
  → validation against IVPT data → closed-loop refinement

This is the top-level API for the entire system.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd

from ..config import load_config, set_seed, get_device
from ..data.skin_sample import SkinSample, SkinSampleCollection
from ..data.drug_properties import DrugDescriptor, DrugLibrary
from ..data.ivpt_data import FranzDiffusionData, IVPTExperiment
from ..atlas.state_space import SkinStateSpace, TissueStateVector
from ..atlas.graphst_wrapper import GraphSTWrapper
from ..atlas.nicheformer_wrapper import NicheformerWrapper
from ..communication.cellchat_wrapper import CellChatWrapper
from ..communication.commot_wrapper import COMMOTWrapper
from ..communication.flowsig_module import FlowSigModule
from ..communication.consensus_modules import ConsensusCrosstalkBuilder
from ..communication.modulation_factors import CommunicationModulationMapper
from ..transport.layered_diffusion import SkinLayerGeometry, LayeredDiffusionPDE, TransportParameters
from ..transport.state_modulation import StateModulationNetwork
from ..transport.drug_transport import DrugTransportPredictor, TransportPrediction
from ..grammar.hypothesis_grammar import SkinBehaviorGrammar
from ..grammar.rule_engine import RuleEngine
from ..grammar.virtual_experiment import CounterfactualSimulator
from ..validation.ivpt_validation import IVPTValidator
from ..validation.perturbation import PerturbationValidator
from ..validation.blind_test import BlindExtrapolationTest
from ..validation.metrics import ValidationMetrics

logger = logging.getLogger(__name__)


class VirtualSkinSolver:
    """Top-level orchestrator for the omics-constrained virtual skin system.

    Usage:
        solver = VirtualSkinSolver()
        solver.build_state_space(adata_st, adata_sc)
        solver.build_communication_modules(adata_st)
        prediction = solver.predict(tissue_state, drug)
        validation = solver.validate(ivpt_data, tissue_state, drug)
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.cfg = load_config(config_path)
        set_seed(self.cfg.get("project", {}).get("seed", 42))
        self.device = get_device(self.cfg) if device is None else __import__("torch").device(device)

        # Geometry
        geo_cfg = self.cfg.get("skin_geometry", {})
        self.geometry = SkinLayerGeometry(
            sc_thickness=geo_cfg.get("stratum_corneum", {}).get("thickness", 15.0),
            ve_thickness=geo_cfg.get("viable_epidermis", {}).get("thickness", 80.0),
            dermis_thickness=geo_cfg.get("dermis", {}).get("thickness", 1200.0),
            appendage_frac=geo_cfg.get("appendage", {}).get("fractional_area", 0.001),
        )

        # Sub-modules
        self.state_space = SkinStateSpace(
            graphst=GraphSTWrapper(
                device=self.device,
                **{k: v for k, v in self.cfg.get("graphst", {}).items()
                   if k in ["learning_rate", "epochs", "dim_output", "alpha", "beta", "n_top_genes"]},
            ),
        )

        self.crosstalk_builder = ConsensusCrosstalkBuilder(
            cellchat=CellChatWrapper(**self.cfg.get("communication", {}).get("cellchat", {})),
            commot=COMMOTWrapper(**self.cfg.get("communication", {}).get("commot", {})),
            flowsig=FlowSigModule(**self.cfg.get("communication", {}).get("flowsig", {})),
        )
        self.modulation_mapper = CommunicationModulationMapper()

        mod_cfg = self.cfg.get("state_modulation", {})
        self.mod_net = StateModulationNetwork(
            hidden_dims=mod_cfg.get("modulation_network", {}).get("hidden_dims", [32, 16]),
        )

        self.grammar = SkinBehaviorGrammar.default_skin_grammar()
        self.rule_engine = RuleEngine(self.grammar)

        self.predictor = DrugTransportPredictor(
            modulation_net=self.mod_net,
            geometry=self.geometry,
            device=self.device,
        )

        self.counterfactual = CounterfactualSimulator(self.predictor, self.rule_engine)

        # State caches
        self._tissue_state: Optional[TissueStateVector] = None
        self._communication_built = False

    # ================================================================
    # Stage 1: Build state space
    # ================================================================

    def build_state_space(
        self,
        adata_st: ad.AnnData,
        adata_sc: Optional[ad.AnnData] = None,
    ) -> TissueStateVector:
        """Construct the tissue state vector from spatial + scRNA data."""
        logger.info("Building skin state space...")
        self._tissue_state = self.state_space.encode_tissue_state(adata_st, adata_sc)
        logger.info(f"Tissue state: {self._tissue_state.to_array()}")
        return self._tissue_state

    # ================================================================
    # Stage 2: Build communication modules
    # ================================================================

    def build_communication_modules(
        self,
        adata: ad.AnnData,
        groupby: str = "cell_type",
    ) -> pd.DataFrame:
        """Run consensus CCC analysis and build transport modulation modules."""
        logger.info("Building consensus crosstalk modules...")
        self.crosstalk_builder.build(adata, groupby=groupby)
        self._communication_built = True
        summary = self.crosstalk_builder.summary()
        logger.info(f"Communication modules:\n{summary}")
        return summary

    # ================================================================
    # Stage 3: Predict
    # ================================================================

    def predict(
        self,
        tissue_state: Optional[TissueStateVector] = None,
        drug: Optional[DrugDescriptor] = None,
        apply_grammar: bool = True,
        apply_communication: bool = True,
        **kwargs: Any,
    ) -> TransportPrediction:
        """Full prediction pipeline: state → modulation → PDE → endpoints."""
        ts = tissue_state or self._tissue_state
        if ts is None:
            raise RuntimeError("Tissue state not available — call build_state_space first.")
        if drug is None:
            raise ValueError("Drug descriptor is required.")

        # Base parameters from modulation network
        base_params = self.mod_net.predict_params(ts, drug.to_vector())

        # Apply communication modulation
        if apply_communication and self._communication_built:
            base_params = self.modulation_mapper.apply_to_params(
                base_params, self.crosstalk_builder
            )

        # Apply grammar rules
        if apply_grammar:
            base_params, audit = self.rule_engine.evaluate(ts, base_params)
            if audit:
                logger.info(f"Grammar rules triggered: {len(audit)}")

        # Solve PDE
        tp = TransportParameters(
            D_sc=base_params.get("D_sc", 1e-4),
            D_ve=base_params.get("D_ve", 1e-2),
            D_dermis=base_params.get("D_ve", 1e-2) * 5,
            K_sc_ve=base_params.get("K_sc_ve", 10.0),
            k_bind_dermis=base_params.get("k_bind_dermis", 0.0),
            k_clear_vasc=base_params.get("k_clear_vasc", 0.01),
            w_appendage=base_params.get("w_appendage", 0.0),
            C_donor=drug.solubility_mg_mL * 1000 if drug.solubility_mg_mL else 1000.0,
        )

        pde = LayeredDiffusionPDE(geometry=self.geometry, params=tp)
        t_total = kwargs.get("t_total_h", self.cfg.get("transport", {}).get("time_total_hours", 48.0))
        dt = kwargs.get("dt_h", self.cfg.get("transport", {}).get("dt_output_hours", 0.5))
        result = pde.solve(
            t_total_s=t_total * 3600,
            dt_output_s=dt * 3600,
            dose_type=kwargs.get("dose_type", "infinite"),
        )

        jss = pde.steady_state_flux(result)
        tlag = pde.lag_time(result)

        dermis_mask = result["layer_ids"] == 2
        dermis_conc = result["concentration_profile"][:, dermis_mask]
        dermis_mean = dermis_conc.mean(axis=1) if dermis_conc.size > 0 else np.zeros(1)

        return TransportPrediction(
            drug_name=drug.name,
            tissue_state=ts,
            time_h=result["time_h"],
            cumulative_permeation=result["cumulative_permeation"],
            flux=result["flux"],
            concentration_profile=result["concentration_profile"],
            steady_state_flux=jss,
            lag_time=tlag,
            layer_retention=result["layer_retention"],
            target_layer_auc=float(np.trapezoid(dermis_mean, result["time_h"])),
            target_layer_cmax=float(dermis_mean.max()),
            transport_params=base_params,
        )

    # ================================================================
    # Stage 4: Validate
    # ================================================================

    def validate_ivpt(
        self,
        experiment: IVPTExperiment,
        tissue_states: List[TissueStateVector],
        drug: DrugDescriptor,
    ) -> Dict[str, float]:
        """Validate predictions against IVPT experimental data."""
        validator = IVPTValidator(self.predictor)
        validator.validate_experiment(experiment, tissue_states, drug)
        return validator.aggregate_metrics()

    def run_blind_test(
        self,
        test_samples: List[Dict[str, Any]],
        holdout_type: str = "unseen_donor",
    ) -> pd.DataFrame:
        """Run blind extrapolation tests."""
        from ..validation.blind_test import HoldoutType
        tester = BlindExtrapolationTest(self.predictor)
        ht = HoldoutType(holdout_type)
        return tester.run_blind_test(test_samples, ht)

    # ================================================================
    # Stage 5: Virtual experiments
    # ================================================================

    def virtual_experiment(
        self,
        drug: DrugDescriptor,
        axis: str,
        target_value: float,
        tissue_state: Optional[TissueStateVector] = None,
    ) -> Dict[str, Any]:
        """Run a virtual perturbation experiment."""
        ts = tissue_state or self._tissue_state
        if ts is None:
            raise RuntimeError("Build state space first.")
        exp = self.counterfactual.perturb_state_axis(ts, drug, axis, target_value)
        return exp.to_dict()

    def suggest_experiment(
        self,
        drug: DrugDescriptor,
        tissue_state: Optional[TissueStateVector] = None,
    ) -> Dict[str, Any]:
        """Suggest the most informative wet-lab perturbation."""
        ts = tissue_state or self._tissue_state
        if ts is None:
            raise RuntimeError("Build state space first.")
        return self.counterfactual.suggest_most_informative_experiment(ts, drug)

    # ================================================================
    # Explanation & audit
    # ================================================================

    def explain_prediction(
        self,
        tissue_state: Optional[TissueStateVector] = None,
        drug: Optional[DrugDescriptor] = None,
    ) -> Dict[str, Any]:
        """Provide a human-readable explanation of how the prediction was made."""
        ts = tissue_state or self._tissue_state
        explanation = {
            "tissue_state": dict(zip(TissueStateVector.axis_names(), ts.to_array().tolist())) if ts else {},
            "active_grammar_rules": [],
            "communication_modulation": [],
        }

        if ts:
            state_dict = dict(zip(TissueStateVector.axis_names(), ts.to_array()))
            triggered = self.grammar.evaluate_all(state_dict)
            explanation["active_grammar_rules"] = [r.to_natural_language() for r in triggered]

        if self._communication_built:
            explanation["communication_modulation"] = (
                self.modulation_mapper.explain_modulation(self.crosstalk_builder)
            )

        return explanation

    def get_state_summary(self) -> Dict[str, Any]:
        """Return current system state summary."""
        return {
            "tissue_state": self._tissue_state.to_array().tolist() if self._tissue_state else None,
            "communication_built": self._communication_built,
            "n_grammar_rules": len(self.grammar.rules),
            "geometry": {
                "sc_thickness": self.geometry.sc_thickness,
                "ve_thickness": self.geometry.ve_thickness,
                "dermis_thickness": self.geometry.dermis_thickness,
            },
        }
