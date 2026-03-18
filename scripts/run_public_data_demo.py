#!/usr/bin/env python3
"""
End-to-end Virtual Skin pipeline demo using public datasets.

This script demonstrates the complete workflow:
  1. Acquire public data (scRNA-seq + spatial + IVPT literature)
  2. Build tissue state space (atlas)
  3. Infer cell-cell communication
  4. Predict drug transport for multiple drugs
  5. Validate against literature IVPT values
  6. Run virtual experiments (counterfactual simulations)

Usage:
    python scripts/run_public_data_demo.py

Output:
    results/public_demo/ — all plots, tables, and intermediate data
"""

import sys
import os
import logging
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("public_demo")

RESULTS_DIR = "results/public_demo"
DATA_DIR = "data/public"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # ═══════════════════════════════════════════════════════════
    # Step 1: Prepare public data
    # ═══════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("STEP 1: Acquiring public datasets")
    logger.info("=" * 70)

    from virtual_skin.data.public_datasets import PublicDataIntegrator

    integrator = PublicDataIntegrator(output_dir=DATA_DIR)
    scrna_path, spatial_path, ivpt_data = integrator.prepare_all(
        scrna_id="GSE147424",
        spatial_id="visium_human_skin_ffpe",
    )

    logger.info(f"scRNA-seq: {scrna_path}")
    logger.info(f"Spatial:   {spatial_path}")
    logger.info(f"IVPT curves: {len(ivpt_data['curves'])}")

    # ═══════════════════════════════════════════════════════════
    # Step 2: Load data and build state space
    # ═══════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Building tissue state space")
    logger.info("=" * 70)

    import anndata as ad

    adata_sc = ad.read_h5ad(scrna_path)
    adata_sp = ad.read_h5ad(spatial_path)

    logger.info(f"scRNA: {adata_sc.n_obs} cells × {adata_sc.n_vars} genes")
    logger.info(f"Spatial: {adata_sp.n_obs} spots × {adata_sp.n_vars} genes")
    logger.info(f"Conditions in scRNA: {adata_sc.obs['condition'].value_counts().to_dict()}")

    from virtual_skin.atlas.layer_state import LayerStateEncoder
    from virtual_skin.atlas.niche_state import NicheStateEncoder
    from virtual_skin.atlas.cell_state import CellStateEncoder
    from virtual_skin.atlas.state_space import SkinStateSpace, TissueStateVector

    # Encode layers, niches, cell states
    layer_enc = LayerStateEncoder()
    niche_enc = NicheStateEncoder()
    cell_enc = CellStateEncoder()

    adata_sc = layer_enc.assign_layers(adata_sc)
    adata_sp = layer_enc.assign_layers(adata_sp)
    if "skin_layer" in adata_sc.obs.columns:
        logger.info(f"Layer assignment (scRNA): {adata_sc.obs['skin_layer'].value_counts().to_dict()}")

    adata_sc = niche_enc.score_niches(adata_sc)
    adata_sp = niche_enc.score_niches(adata_sp)
    if "dominant_niche" in adata_sc.obs.columns:
        logger.info(f"Niche assignment: {adata_sc.obs['dominant_niche'].value_counts().to_dict()}")

    adata_sc = cell_enc.score_all_programs(adata_sc)

    # Build state space and compute per-condition state vectors
    state_space = SkinStateSpace(layer_enc=layer_enc, niche_enc=niche_enc, cell_enc=cell_enc)
    state_vectors = {}

    for condition in adata_sc.obs["condition"].unique():
        mask = adata_sc.obs["condition"] == condition
        adata_cond = adata_sc[mask].copy()
        try:
            sv = state_space.encode_tissue_state(adata_sp, adata_sc=adata_cond)
        except Exception as e:
            logger.warning(f"State encoding failed for {condition}: {e}")
            # Fallback with manual defaults based on condition type
            if "atopic" in str(condition).lower():
                sv = TissueStateVector(barrier_integrity=0.3, inflammatory_load=0.7,
                                        ecm_remodeling=0.4, vascularization=0.6, appendage_openness=0.4)
            elif "psoriasis" in str(condition).lower():
                sv = TissueStateVector(barrier_integrity=0.4, inflammatory_load=0.6,
                                        ecm_remodeling=0.5, vascularization=0.7, appendage_openness=0.3)
            else:
                sv = TissueStateVector(barrier_integrity=0.8, inflammatory_load=0.15,
                                        ecm_remodeling=0.2, vascularization=0.5, appendage_openness=0.3)
        state_vectors[condition] = sv
        logger.info(f"State vector [{condition}]: {sv}")

    # ═══════════════════════════════════════════════════════════
    # Step 3: Cell-cell communication analysis
    # ═══════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Cell-cell communication analysis")
    logger.info("=" * 70)

    from virtual_skin.communication.consensus_modules import (
        ConsensusCrosstalkBuilder, CrosstalkModule, SKIN_MODULE_DEFINITIONS,
    )
    from virtual_skin.communication.modulation_factors import CommunicationModulationMapper

    consensus_builder = ConsensusCrosstalkBuilder()

    # Try full CCC pipeline; fall back to synthetic modules if deps missing
    try:
        consensus_builder.build(adata_sc, groupby="cell_type")
        logger.info(f"Consensus modules (full CCC): {list(consensus_builder.modules.keys())}")
    except Exception as e:
        logger.warning(f"Full CCC pipeline unavailable ({e}). Using synthetic modules.")
        for mod_name, mod_def in SKIN_MODULE_DEFINITIONS.items():
            m = CrosstalkModule(name=mod_name, description=mod_def["description"])
            m.activity = np.random.uniform(0.2, 0.6)
            m.consensus_confidence = 0.5
            consensus_builder.modules[mod_name] = m
        logger.info(f"Synthetic consensus modules: {list(consensus_builder.modules.keys())}")

    for name, mod in consensus_builder.modules.items():
        logger.info(f"  {name:30s} | activity={mod.activity:.3f} | confidence={mod.consensus_confidence:.2f}")

    # Map to transport modulation
    mod_mapper = CommunicationModulationMapper()
    modulation_factors = mod_mapper.compute_modulation_factors(consensus_builder)
    logger.info(f"Modulation factors: {modulation_factors}")

    # ═══════════════════════════════════════════════════════════
    # Step 4: Multi-drug transport prediction
    # ═══════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Drug transport prediction (multi-drug × multi-state)")
    logger.info("=" * 70)

    from virtual_skin.transport.drug_transport import DrugTransportPredictor
    from virtual_skin.data.drug_properties import DrugLibrary, DrugDescriptor

    drug_lib = DrugLibrary.default_library()
    predictor = DrugTransportPredictor(use_default_physics=True)

    test_drugs = ["caffeine", "hydrocortisone", "testosterone", "lidocaine", "diclofenac"]
    prediction_results = {}

    for drug_name in test_drugs:
        try:
            drug = drug_lib.get(drug_name)
        except KeyError:
            db = ivpt_data["db"]
            records = db.get_drug_records(drug_name)
            if records:
                r = records[0]
                drug = DrugDescriptor(
                    name=drug_name, molecular_weight=r.mw, logP=r.logp,
                    solubility_mg_mL=r.donor_conc_mg_ml or 10.0,
                )
            else:
                logger.warning(f"Skipping {drug_name}: no drug descriptor")
                continue

        for condition, sv in state_vectors.items():
            try:
                result = predictor.predict(
                    tissue_state=sv,
                    drug=drug,
                    t_total_h=24.0,
                )
                key = f"{drug_name}__{condition}"
                prediction_results[key] = result
                logger.info(
                    f"  {drug_name:20s} | {condition:30s} | "
                    f"Jss={result.steady_state_flux:.3f} µg/cm²/h | "
                    f"Q24={result.cumulative_permeation[-1]:.2f} µg/cm²"
                )
            except Exception as e:
                logger.warning(f"  {drug_name} × {condition} failed: {e}")

    # ═══════════════════════════════════════════════════════════
    # Step 5: Validate against literature IVPT
    # ═══════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: Validation against literature IVPT data")
    logger.info("=" * 70)

    from virtual_skin.data.public_datasets.ivpt_literature import IVPTLiteratureDB

    db = IVPTLiteratureDB()
    validation_rows = []

    for drug_name in test_drugs:
        pred_key = f"{drug_name}__healthy"
        if pred_key not in prediction_results:
            continue

        pred = prediction_results[pred_key]
        lit_jss = db.get_consensus_jss(drug_name)
        lit_lag = db.get_consensus_lag_time(drug_name)
        pred_jss = pred.steady_state_flux

        if lit_jss and lit_jss > 0:
            fold_error = max(pred_jss / max(lit_jss, 1e-12), lit_jss / max(pred_jss, 1e-12))
        else:
            fold_error = float("nan")

        validation_rows.append({
            "drug": drug_name,
            "predicted_jss": pred_jss,
            "literature_jss": lit_jss,
            "fold_error": fold_error,
            "predicted_Q24": float(pred.cumulative_permeation[-1]),
            "literature_lag_h": lit_lag,
        })
        logger.info(
            f"  {drug_name:20s} | pred Jss={pred_jss:.3f} | lit Jss={lit_jss:.3f} | "
            f"fold error={fold_error:.1f}x"
        )

    val_df = pd.DataFrame(validation_rows)
    val_path = os.path.join(RESULTS_DIR, "validation_vs_literature.csv")
    val_df.to_csv(val_path, index=False)
    logger.info(f"Validation table saved to {val_path}")

    # ═══════════════════════════════════════════════════════════
    # Step 6: Virtual experiments
    # ═══════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6: Virtual experiments (counterfactual simulations)")
    logger.info("=" * 70)

    from virtual_skin.grammar.hypothesis_grammar import SkinBehaviorGrammar
    from virtual_skin.grammar.rule_engine import RuleEngine

    grammar = SkinBehaviorGrammar()
    rule_engine = RuleEngine(grammar)

    # Perturbation: What if barrier integrity is reduced (AD-like)?
    baseline_sv = state_vectors.get("healthy",
        TissueStateVector(barrier_integrity=0.8, inflammatory_load=0.15,
                          ecm_remodeling=0.2, vascularization=0.5, appendage_openness=0.3))

    caff = drug_lib.get("caffeine")

    perturbations = {
        "barrier_disruption_50pct": {"barrier_integrity": -0.4},
        "inflammation_high": {"inflammatory_load": +0.5},
        "enhanced_vascularization": {"vascularization": +0.3},
    }

    for exp_name, deltas in perturbations.items():
        base_arr = baseline_sv.to_array().copy()
        axis_names = TissueStateVector.axis_names()
        for axis, delta in deltas.items():
            if axis in axis_names:
                idx = axis_names.index(axis)
                base_arr[idx] = np.clip(base_arr[idx] + delta, 0, 1)

        perturbed_sv = TissueStateVector(
            barrier_integrity=float(base_arr[0]), inflammatory_load=float(base_arr[1]),
            ecm_remodeling=float(base_arr[2]), vascularization=float(base_arr[3]),
            appendage_openness=float(base_arr[4]),
        )

        try:
            base_result = predictor.predict(tissue_state=baseline_sv, drug=caff, t_total_h=24.0)
            pert_result = predictor.predict(tissue_state=perturbed_sv, drug=caff, t_total_h=24.0)
            ratio = pert_result.steady_state_flux / max(base_result.steady_state_flux, 1e-10)
            logger.info(f"  {exp_name:35s} | Jss ratio vs baseline: {ratio:.2f}x")
        except Exception as e:
            logger.warning(f"  {exp_name} failed: {e}")

    # ═══════════════════════════════════════════════════════════
    # Step 7: Generate visualizations
    # ═══════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("STEP 7: Generating visualizations")
    logger.info("=" * 70)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # --- Plot 1: Drug permeation comparison ---
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, drug_name in enumerate(test_drugs[:5]):
            ax = axes[idx]
            for condition in ["healthy", "atopic_dermatitis"]:
                key = f"{drug_name}__{condition}"
                if key in prediction_results:
                    res = prediction_results[key]
                    label = condition.replace("_", " ").title()
                    ax.plot(res.time_h, res.cumulative_permeation, label=label, linewidth=2)

            # Overlay literature reference
            lit_curve = db.generate_ivpt_curve(drug_name, total_time_h=24.0)
            ax.plot(lit_curve["time_h"], lit_curve["cumulative_ug_cm2"],
                    "--", color="gray", alpha=0.7, label="Literature ref")

            ax.set_title(drug_name.replace("_", " ").title(), fontsize=13)
            ax.set_xlabel("Time (h)")
            ax.set_ylabel("Cumulative (µg/cm²)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[-1].axis("off")
        plt.suptitle("Multi-Drug Permeation: Model vs Literature", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "permeation_comparison.png"), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved permeation_comparison.png")

        # --- Plot 2: Validation scatter (pred vs lit Jss) ---
        if len(val_df) > 0 and val_df["literature_jss"].notna().any():
            fig, ax = plt.subplots(figsize=(8, 8))
            valid = val_df.dropna(subset=["literature_jss", "predicted_jss"])
            ax.scatter(valid["literature_jss"], valid["predicted_jss"], s=100, c="#2196F3", edgecolors="k")
            for _, row in valid.iterrows():
                ax.annotate(row["drug"], (row["literature_jss"], row["predicted_jss"]),
                            fontsize=10, ha="left", va="bottom")

            lims = [0, max(valid["literature_jss"].max(), valid["predicted_jss"].max()) * 1.2]
            ax.plot(lims, lims, "k--", alpha=0.5, label="Perfect agreement")
            ax.plot(lims, [x * 3 for x in lims], "r--", alpha=0.3, label="3-fold")
            ax.plot(lims, [x / 3 for x in lims], "r--", alpha=0.3)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_xlabel("Literature Jss (µg/cm²/h)", fontsize=13)
            ax.set_ylabel("Predicted Jss (µg/cm²/h)", fontsize=13)
            ax.set_title("Model vs Literature: Steady-State Flux", fontsize=15)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, "jss_validation_scatter.png"), dpi=150)
            plt.close()
            logger.info("Saved jss_validation_scatter.png")

        # --- Plot 3: State vectors heatmap ---
        if state_vectors:
            fig, ax = plt.subplots(figsize=(10, 4))
            sv_mat = np.array([state_vectors[c].to_array() for c in state_vectors])
            im = ax.imshow(sv_mat, cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=1)
            ax.set_yticks(range(len(state_vectors)))
            ax.set_yticklabels([c.replace("_", " ").title() for c in state_vectors])
            ax.set_xticks(range(5))
            ax.set_xticklabels(["Barrier\nIntegrity", "Inflammatory\nLoad", "ECM\nRemodeling",
                                "Vascular-\nization", "Appendage\nOpenness"], fontsize=10)
            plt.colorbar(im, ax=ax, label="State score")
            ax.set_title("Tissue State Vectors by Condition", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, "state_vectors_heatmap.png"), dpi=150)
            plt.close()
            logger.info("Saved state_vectors_heatmap.png")

        # --- Plot 4: IVPT database overview ---
        db_df = db.to_dataframe()
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        ax = axes[0]
        drug_counts = db_df["drug"].value_counts()
        ax.barh(range(len(drug_counts)), drug_counts.values, color="#4CAF50")
        ax.set_yticks(range(len(drug_counts)))
        ax.set_yticklabels(drug_counts.index, fontsize=9)
        ax.set_xlabel("Number of records")
        ax.set_title("Records per drug")

        ax = axes[1]
        valid_kp = db_df.dropna(subset=["kp_cm_h"])
        ax.scatter(valid_kp["logp"], np.log10(valid_kp["kp_cm_h"]), c="#FF5722", s=60, edgecolors="k")
        for _, row in valid_kp.iterrows():
            ax.annotate(row["drug"][:6], (row["logp"], np.log10(row["kp_cm_h"])), fontsize=7)
        logp_range = np.linspace(-2, 5, 50)
        mw_mean = valid_kp["mw"].mean()
        potts_guy = -2.7 + 0.71 * logp_range - 0.0061 * mw_mean + np.log10(3600)
        ax.plot(logp_range, potts_guy, "b--", alpha=0.5, label=f"Potts-Guy (MW={mw_mean:.0f})")
        ax.set_xlabel("log P")
        ax.set_ylabel("log₁₀ Kp (cm/h)")
        ax.set_title("Permeability vs Lipophilicity")
        ax.legend()

        ax = axes[2]
        valid_flux = db_df.dropna(subset=["jss_ug_cm2_h"])
        ax.bar(range(len(valid_flux)), valid_flux["jss_ug_cm2_h"].values, color="#2196F3")
        ax.set_xticks(range(len(valid_flux)))
        ax.set_xticklabels(valid_flux["drug"].values, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Jss (µg/cm²/h)")
        ax.set_title("Literature Steady-State Flux")

        plt.suptitle("IVPT Literature Database Overview", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "ivpt_database_overview.png"), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved ivpt_database_overview.png")

    except ImportError:
        logger.warning("Matplotlib not available — skipping plots")

    # ═══════════════════════════════════════════════════════════
    # Final Summary
    # ═══════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)

    logger.info(f"\nResults saved to: {RESULTS_DIR}/")
    logger.info("Files generated:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        size = os.path.getsize(os.path.join(RESULTS_DIR, f))
        logger.info(f"  {f:45s} ({size:>8,} bytes)")

    if len(val_df) > 0:
        logger.info("\nValidation Summary (Predicted vs Literature Jss):")
        logger.info(val_df.to_string(index=False))
        mean_fold = val_df["fold_error"].dropna().mean()
        logger.info(f"\nMean fold error: {mean_fold:.1f}x")
        n_within_3fold = (val_df["fold_error"].dropna() <= 3).sum()
        logger.info(f"Within 3-fold: {n_within_3fold}/{len(val_df)} drugs")

    logger.info("\n📋 Next steps for Nature-quality publication:")
    logger.info("  1. Replace synthetic scRNA with real GSE147424 data (download from GEO)")
    logger.info("  2. Download 10x Visium skin spatial data for real spatial analysis")
    logger.info("  3. Train state modulation network on IVPT calibration set")
    logger.info("  4. Train PINN solver for physics-informed refinement")
    logger.info("  5. Run full Bayesian inference for uncertainty quantification")
    logger.info("  6. Add your private data for novel biological insights")


if __name__ == "__main__":
    main()
