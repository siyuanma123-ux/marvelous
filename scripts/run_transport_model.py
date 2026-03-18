#!/usr/bin/env python3
"""
Script: Run transport model predictions for multiple drugs.

Usage:
    python run_transport_model.py [--drug caffeine] [--state 0.8,0.1,0.2,0.5,0.1]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run Transport Model")
    parser.add_argument("--drug", type=str, default="all",
                        help="Drug name or 'all' for entire library")
    parser.add_argument("--state", type=str, default="0.8,0.1,0.2,0.5,0.1",
                        help="Tissue state vector (5 comma-separated values)")
    parser.add_argument("--output", type=str, default="transport_results.csv")
    args = parser.parse_args()

    from virtual_skin.config import load_config, set_seed
    cfg = load_config()
    set_seed(cfg["project"]["seed"])

    from virtual_skin.atlas.state_space import TissueStateVector
    from virtual_skin.data.drug_properties import DrugLibrary
    from virtual_skin.transport.drug_transport import DrugTransportPredictor
    from virtual_skin.transport.state_modulation import StateModulationNetwork
    from virtual_skin.transport.layered_diffusion import SkinLayerGeometry

    # Parse state
    state_vals = [float(x) for x in args.state.split(",")]
    tissue_state = TissueStateVector(*state_vals)

    # Setup predictor
    geometry = SkinLayerGeometry()
    mod_net = StateModulationNetwork()
    predictor = DrugTransportPredictor(modulation_net=mod_net, geometry=geometry)

    # Load drugs
    drug_lib = DrugLibrary.default_library()
    if args.drug == "all":
        drug_names = drug_lib.list_drugs()
    else:
        drug_names = [args.drug]

    results = []
    for name in drug_names:
        drug = drug_lib.get(name)
        pred = predictor.predict(tissue_state, drug)
        results.append({
            "drug": name,
            "MW": drug.molecular_weight,
            "logP": drug.logP,
            "Jss": pred.steady_state_flux,
            "lag_time": pred.lag_time,
            "Q_48h": float(pred.cumulative_permeation[-1]),
            "target_AUC": pred.target_layer_auc,
            "target_Cmax": pred.target_layer_cmax,
        })
        logger.info(f"  {name}: Jss={pred.steady_state_flux:.4f}, tlag={pred.lag_time:.2f}")

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    logger.info(f"\nResults saved to {args.output}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
