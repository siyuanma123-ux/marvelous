#!/usr/bin/env python3
"""
End-to-end Virtual Skin demonstration.

This script demonstrates the full pipeline using synthetic data,
showing how each module connects in the integrated system.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def create_synthetic_ivpt_data():
    """Generate synthetic IVPT data for demonstration."""
    from virtual_skin.data.ivpt_data import FranzDiffusionData, IVPTExperiment

    np.random.seed(42)
    t = np.linspace(0, 48, 97)  # 0.5h intervals

    experiment = IVPTExperiment("caffeine", "aqueous_solution")

    for i in range(5):
        jss = 5.0 + np.random.normal(0, 0.5)
        tlag = 2.0 + np.random.normal(0, 0.3)
        Q = np.maximum(0, jss * np.maximum(0, t - tlag))
        Q += np.random.normal(0, 0.5, size=Q.shape)
        Q = np.maximum(0, np.cumsum(np.maximum(0, np.diff(np.concatenate([[0], Q])))))

        run = FranzDiffusionData(
            sample_id=f"donor_{i}_forearm_healthy",
            drug_name="caffeine",
            formulation="aqueous_solution",
            time_h=t,
            cumulative_permeation=Q,
            sc_retention=1.5 + np.random.normal(0, 0.2),
            ve_retention=0.8 + np.random.normal(0, 0.1),
            dermis_retention=2.0 + np.random.normal(0, 0.3),
        )
        experiment.add_run(run)

    return experiment


def main():
    logger.info("=" * 70)
    logger.info("  Omics-Constrained Multi-Scale Virtual Skin — Demo Pipeline")
    logger.info("=" * 70)

    # ----------------------------------------------------------------
    # Step 1: Configure
    # ----------------------------------------------------------------
    from virtual_skin.config import load_config, set_seed
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    logger.info("Configuration loaded.")

    # ----------------------------------------------------------------
    # Step 2: Prepare drug library
    # ----------------------------------------------------------------
    from virtual_skin.data.drug_properties import DrugLibrary
    drug_lib = DrugLibrary.default_library()
    caffeine = drug_lib.get("caffeine")
    logger.info(f"Drug: {caffeine.name}, MW={caffeine.molecular_weight}, logP={caffeine.logP}")

    # ----------------------------------------------------------------
    # Step 3: Create tissue state (normally from omics, here synthetic)
    # ----------------------------------------------------------------
    from virtual_skin.atlas.state_space import TissueStateVector
    tissue_state = TissueStateVector(
        barrier_integrity=0.8,
        inflammatory_load=0.1,
        ecm_remodeling=0.2,
        vascularization=0.5,
        appendage_openness=0.1,
    )
    logger.info(f"Tissue state: {tissue_state.to_array()}")

    # ----------------------------------------------------------------
    # Step 4: Initialize grammar rules
    # ----------------------------------------------------------------
    from virtual_skin.grammar.hypothesis_grammar import SkinBehaviorGrammar
    grammar = SkinBehaviorGrammar.default_skin_grammar()
    logger.info(f"Grammar rules loaded: {len(grammar.rules)}")
    for rule_text in grammar.list_rules():
        logger.info(f"  {rule_text}")

    # ----------------------------------------------------------------
    # Step 5: Transport prediction
    # ----------------------------------------------------------------
    from virtual_skin.transport.layered_diffusion import (
        SkinLayerGeometry, LayeredDiffusionPDE, TransportParameters,
    )
    from virtual_skin.transport.drug_transport import DrugTransportPredictor
    from virtual_skin.transport.state_modulation import StateModulationNetwork

    geometry = SkinLayerGeometry()
    mod_net = StateModulationNetwork()
    predictor = DrugTransportPredictor(
        modulation_net=mod_net,
        geometry=geometry,
    )

    prediction = predictor.predict(tissue_state, caffeine)
    logger.info(f"\n--- Prediction Results ---")
    logger.info(f"  Steady-state flux (Jss): {prediction.steady_state_flux:.4f} µg/cm²/h")
    logger.info(f"  Lag time: {prediction.lag_time:.2f} h")
    logger.info(f"  Total permeation (48h): {prediction.cumulative_permeation[-1]:.2f} µg/cm²")
    logger.info(f"  Target-layer AUC: {prediction.target_layer_auc:.4f}")
    logger.info(f"  Target-layer Cmax: {prediction.target_layer_cmax:.4f}")
    logger.info(f"  Layer retention: {prediction.layer_retention}")

    # ----------------------------------------------------------------
    # Step 6: Sensitivity analysis
    # ----------------------------------------------------------------
    logger.info("\n--- Sensitivity Analysis ---")
    for axis in TissueStateVector.axis_names():
        sweep = predictor.sensitivity_analysis(tissue_state, caffeine, axis=axis, n_points=5)
        jss_range = sweep["Jss"]
        logger.info(
            f"  {axis}: Jss range = [{jss_range.min():.4f}, {jss_range.max():.4f}]"
        )

    # ----------------------------------------------------------------
    # Step 7: Virtual experiment (counterfactual)
    # ----------------------------------------------------------------
    from virtual_skin.grammar.virtual_experiment import CounterfactualSimulator
    from virtual_skin.grammar.rule_engine import RuleEngine

    rule_engine = RuleEngine(grammar)
    cf_sim = CounterfactualSimulator(predictor, rule_engine)

    exp = cf_sim.perturb_state_axis(
        tissue_state, caffeine,
        axis="inflammatory_load", target_value=0.8,
        experiment_id="VE_inflammation_high",
    )
    logger.info(f"\n--- Virtual Experiment: High Inflammation ---")
    logger.info(f"  ΔJss: {exp.delta_flux:.4f}")
    logger.info(f"  Δlag_time: {exp.delta_lag_time:.4f}")
    logger.info(f"  ΔAUC: {exp.delta_auc:.4f}")

    # ----------------------------------------------------------------
    # Step 8: Validation against synthetic IVPT
    # ----------------------------------------------------------------
    from virtual_skin.validation.ivpt_validation import IVPTValidator

    experiment = create_synthetic_ivpt_data()
    validator = IVPTValidator(predictor)
    for run in experiment.runs:
        validator.validate_single(run, tissue_state, caffeine)

    metrics = validator.aggregate_metrics()
    logger.info(f"\n--- IVPT Validation Metrics ---")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # ----------------------------------------------------------------
    # Step 9: Grammar rule audit
    # ----------------------------------------------------------------
    base_params = mod_net.predict_params(tissue_state, caffeine.to_vector())
    modulated, audit = rule_engine.evaluate(tissue_state, base_params)
    logger.info(f"\n--- Grammar Rule Audit ---")
    logger.info(f"  Rules triggered: {len(audit)}")
    for entry in audit:
        logger.info(f"    {entry['rule_name']}: {entry['param']} "
                     f"{entry['old_value']:.6f} → {entry['new_value']:.6f}")

    logger.info("\n" + "=" * 70)
    logger.info("  Virtual Skin Demo Complete")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
