#!/usr/bin/env python3
"""
Script: Run cell communication analysis and build consensus modules.

Usage:
    python run_communication_analysis.py --adata <path.h5ad> --groupby cell_type
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Cell Communication Analysis")
    parser.add_argument("--adata", type=str, help="Path to AnnData (.h5ad)")
    parser.add_argument("--groupby", type=str, default="cell_type")
    parser.add_argument("--output", type=str, default="communication_modules.csv")
    args = parser.parse_args()

    from virtual_skin.config import load_config, set_seed
    cfg = load_config()
    set_seed(cfg["project"]["seed"])

    if args.adata:
        import scanpy as sc
        adata = sc.read_h5ad(args.adata)
        logger.info(f"Loaded {adata.n_obs} cells, {adata.n_vars} genes")

        from virtual_skin.communication.consensus_modules import ConsensusCrosstalkBuilder
        builder = ConsensusCrosstalkBuilder()
        modules = builder.build(adata, groupby=args.groupby)
        summary = builder.summary()
    else:
        logger.info("No data provided — showing default module definitions")
        from virtual_skin.communication.consensus_modules import SKIN_MODULE_DEFINITIONS
        import pandas as pd
        rows = []
        for name, defn in SKIN_MODULE_DEFINITIONS.items():
            rows.append({
                "module": name,
                "description": defn["description"],
                "target_param": defn["target_param"],
                "direction": defn["direction"],
                "keywords": ", ".join(defn["pathway_keywords"]),
            })
        summary = pd.DataFrame(rows)

    summary.to_csv(args.output, index=False)
    logger.info(f"\nModule summary saved to {args.output}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
