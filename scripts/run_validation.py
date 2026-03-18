#!/usr/bin/env python3
"""
Script: Run validation and blind-testing pipeline.

Usage:
    python run_validation.py [--output validation_report.csv]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Validation Pipeline")
    parser.add_argument("--output", type=str, default="validation_report.csv")
    args = parser.parse_args()

    from virtual_skin.config import load_config, set_seed
    cfg = load_config()
    set_seed(cfg["project"]["seed"])

    from virtual_skin.atlas.state_space import TissueStateVector
    from virtual_skin.data.drug_properties import DrugLibrary
    from virtual_skin.data.ivpt_data import FranzDiffusionData, IVPTExperiment
    from virtual_skin.transport.drug_transport import DrugTransportPredictor
    from virtual_skin.transport.state_modulation import StateModulationNetwork
    from virtual_skin.transport.layered_diffusion import SkinLayerGeometry
    from virtual_skin.validation.ivpt_validation import IVPTValidator
    from virtual_skin.validation.metrics import ValidationMetrics

    # Setup
    geometry = SkinLayerGeometry()
    mod_net = StateModulationNetwork()
    predictor = DrugTransportPredictor(modulation_net=mod_net, geometry=geometry)
    drug_lib = DrugLibrary.default_library()
    caffeine = drug_lib.get("caffeine")

    # Generate synthetic IVPT data for demonstration
    logger.info("Generating synthetic IVPT validation data...")
    conditions = [
        ("healthy", TissueStateVector(0.8, 0.1, 0.2, 0.5, 0.1)),
        ("inflamed", TissueStateVector(0.4, 0.7, 0.3, 0.6, 0.1)),
        ("fibrotic", TissueStateVector(0.6, 0.2, 0.8, 0.4, 0.05)),
    ]

    validator = IVPTValidator(predictor)
    np.random.seed(42)

    for cond_name, ts in conditions:
        pred = predictor.predict(ts, caffeine)
        for rep in range(3):
            noise = np.random.normal(1.0, 0.15)
            obs_jss = pred.steady_state_flux * noise
            t = pred.time_h
            Q_obs = pred.cumulative_permeation * noise + np.random.normal(0, 0.1, len(t))
            Q_obs = np.maximum(0, Q_obs)

            run = FranzDiffusionData(
                sample_id=f"{cond_name}_rep{rep}",
                drug_name="caffeine",
                formulation="aqueous",
                time_h=t,
                cumulative_permeation=Q_obs,
                sc_retention=pred.layer_retention.get("sc", 0) * noise,
                dermis_retention=pred.layer_retention.get("dermis", 0) * noise,
            )
            validator.validate_single(run, ts, caffeine)

    summary = validator.summary()
    metrics = validator.aggregate_metrics()

    logger.info("\n--- Validation Summary ---")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    summary.to_csv(args.output, index=False)
    logger.info(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
