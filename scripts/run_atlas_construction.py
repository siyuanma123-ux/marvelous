#!/usr/bin/env python3
"""
Script: Build skin state atlas from multi-modal omics data.

Usage:
    python run_atlas_construction.py --sc_path <path> --st_path <path> [--output <path>]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build Skin State Atlas")
    parser.add_argument("--sc_path", type=str, help="Path to scRNA-seq data (.h5ad)")
    parser.add_argument("--st_path", type=str, help="Path to spatial transcriptomics data (.h5ad)")
    parser.add_argument("--output", type=str, default="skin_atlas_state.npz",
                        help="Output path for tissue state vector")
    parser.add_argument("--n_clusters", type=int, default=7,
                        help="Number of spatial clusters")
    args = parser.parse_args()

    from virtual_skin.config import load_config, set_seed
    cfg = load_config()
    set_seed(cfg["project"]["seed"])

    from virtual_skin.atlas.state_space import SkinStateSpace
    from virtual_skin.data.omics_loader import load_scrna, load_spatial

    state_space = SkinStateSpace()

    if args.sc_path and args.st_path:
        logger.info(f"Loading scRNA-seq from {args.sc_path}")
        adata_sc = load_scrna(args.sc_path)
        logger.info(f"  {adata_sc.n_obs} cells, {adata_sc.n_vars} genes")

        logger.info(f"Loading spatial data from {args.st_path}")
        adata_st = load_spatial(args.st_path)
        logger.info(f"  {adata_st.n_obs} spots, {adata_st.n_vars} genes")

        logger.info("Building tissue state...")
        tissue_state = state_space.encode_tissue_state(adata_st, adata_sc)
    else:
        logger.info("No data paths provided — running with synthetic example")
        from virtual_skin.atlas.state_space import TissueStateVector
        tissue_state = TissueStateVector(
            barrier_integrity=0.75,
            inflammatory_load=0.15,
            ecm_remodeling=0.20,
            vascularization=0.45,
            appendage_openness=0.10,
        )

    import numpy as np
    np.savez(args.output, state_vector=tissue_state.to_array(),
             axis_names=tissue_state.axis_names())
    logger.info(f"Tissue state saved to {args.output}")
    logger.info(f"State vector: {tissue_state.to_array()}")


if __name__ == "__main__":
    main()
