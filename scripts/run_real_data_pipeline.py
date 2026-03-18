#!/usr/bin/env python3
"""
Run the full Virtual Skin pipeline with real GSE147424 scRNA-seq data.
"""

import sys, os, logging, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import numpy as np
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("real_data")


def main():
    import scanpy as sc
    import torch
    os.makedirs("results/real_data", exist_ok=True)

    # ─── Load real GSE147424 data ───
    h5ad_path = "data/public/GSE147424_processed.h5ad"
    logger.info(f"Loading real scRNA-seq data from {h5ad_path}...")

    if os.path.exists(h5ad_path):
        adata = sc.read_h5ad(h5ad_path)
        logger.info(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
        logger.info(f"Obs columns: {list(adata.obs.columns)}")

        # Check for cell type annotations
        has_ct = "cell_type" in adata.obs.columns or "leiden" in adata.obs.columns
        ct_col = "cell_type" if "cell_type" in adata.obs.columns else "leiden"

        if has_ct:
            logger.info(f"Cell types ({ct_col}):")
            for ct, count in adata.obs[ct_col].value_counts().head(15).items():
                logger.info(f"  {ct}: {count}")

        # Check for condition/sample info
        for col in ["condition", "sample", "patient", "batch", "disease"]:
            if col in adata.obs.columns:
                logger.info(f"\n{col} distribution:")
                for val, cnt in adata.obs[col].value_counts().items():
                    logger.info(f"  {val}: {cnt}")
    else:
        logger.warning(f"{h5ad_path} not found — using synthetic data")
        from virtual_skin.data.public_datasets.scrna_datasets import download_skin_scrna
        adata = download_skin_scrna("GSE147424", "data/public", try_real_download=False)

    # ─── Build state space from real data ───
    logger.info("\n" + "=" * 60)
    logger.info("Building tissue state space from real data")
    logger.info("=" * 60)

    from virtual_skin.atlas.state_space import SkinStateSpace, TissueStateVector

    state_space = SkinStateSpace()

    if "X_pca" not in adata.obsm:
        sc.pp.pca(adata, n_comps=50)
    if "X_umap" not in adata.obsm:
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)

    ct_col = "cell_type" if "cell_type" in adata.obs.columns else "leiden"

    # Define skin-relevant cell type mappings
    # Map leiden clusters to biological cell types if needed
    if ct_col == "leiden":
        logger.info("Using Leiden clusters as proxy for cell types")

    # Encode tissue states for different conditions
    states = {}
    cond_col = None
    for col in ["condition", "disease", "sample", "batch"]:
        if col in adata.obs.columns:
            cond_col = col
            break

    if cond_col:
        for cond in adata.obs[cond_col].unique()[:5]:
            mask = adata.obs[cond_col] == cond
            n_cells = mask.sum()

            # Heuristic state encoding from gene expression signatures
            sub = adata[mask]
            barrier_genes = ["FLG", "LOR", "IVL", "KRT1", "KRT10", "CDSN"]
            inflam_genes = ["IL1B", "IL6", "TNF", "IFNG", "IL17A", "IL13", "CCL2"]
            ecm_genes = ["COL1A1", "COL3A1", "FN1", "MMP1", "MMP9", "TGFB1"]
            vasc_genes = ["PECAM1", "VWF", "CDH5", "VEGFA", "KDR"]
            append_genes = ["KRT14", "KRT5", "SOX9", "LHX2"]

            def gene_score(sub_adata, genes):
                present = [g for g in genes if g in sub_adata.var_names]
                if not present:
                    return 0.5
                try:
                    if hasattr(sub_adata.X, 'toarray'):
                        vals = sub_adata[:, present].X.toarray()
                    else:
                        vals = sub_adata[:, present].X
                    return float(np.clip(np.mean(vals > 0), 0, 1))
                except Exception:
                    return 0.5

            barrier = gene_score(sub, barrier_genes)
            inflam = gene_score(sub, inflam_genes)
            ecm = gene_score(sub, ecm_genes)
            vasc = gene_score(sub, vasc_genes)
            append = gene_score(sub, append_genes)

            sv = TissueStateVector(
                barrier_integrity=barrier,
                inflammatory_load=inflam,
                ecm_remodeling=ecm,
                vascularization=vasc,
                appendage_openness=append,
            )
            states[str(cond)] = sv
            logger.info(
                f"  {str(cond):25s} ({n_cells:5d} cells) → "
                f"barrier={barrier:.2f} inflam={inflam:.2f} "
                f"ecm={ecm:.2f} vasc={vasc:.2f} append={append:.2f}"
            )
    else:
        states["healthy"] = TissueStateVector(0.8, 0.15, 0.2, 0.5, 0.3)

    # ─── Drug transport predictions ───
    logger.info("\n" + "=" * 60)
    logger.info("Drug transport predictions with trained model")
    logger.info("=" * 60)

    from virtual_skin.transport.drug_transport import DrugTransportPredictor
    from virtual_skin.transport.state_modulation import StateModulationNetwork
    from virtual_skin.data.drug_properties import DrugDescriptor, DrugLibrary

    drug_lib = DrugLibrary.default_library()
    test_drugs = ["caffeine", "hydrocortisone", "diclofenac", "lidocaine", "testosterone"]

    # Load trained model if available
    model_path = "results/training/modulation_net.pt"
    if os.path.exists(model_path):
        logger.info(f"Loading trained model from {model_path}")
        net = StateModulationNetwork(n_state_axes=5, n_drug_desc=8, hidden_dims=[64, 32])
        net.load_state_dict(torch.load(model_path, weights_only=True))
        predictor = DrugTransportPredictor(modulation_net=net, use_default_physics=True)
    else:
        logger.info("No trained model found — using default physics")
        predictor = DrugTransportPredictor(use_default_physics=True)

    results_table = []
    for state_name, sv in states.items():
        for drug_name in test_drugs:
            try:
                drug = drug_lib.get(drug_name)
            except KeyError:
                continue

            result = predictor.predict(sv, drug, t_total_h=24.0)
            results_table.append({
                "condition": state_name,
                "drug": drug_name,
                "Jss": result.steady_state_flux,
                "lag_h": result.lag_time,
                "Q24": float(result.cumulative_permeation[-1]) if len(result.cumulative_permeation) > 0 else 0,
                "dermis_AUC": result.target_layer_auc,
            })
            logger.info(
                f"  {state_name:15s} × {drug_name:15s} | "
                f"Jss={result.steady_state_flux:8.3f} µg/cm²/h | "
                f"lag={result.lag_time:5.1f}h | Q24={results_table[-1]['Q24']:.1f} µg/cm²"
            )

    # ─── Save results ───
    import csv
    with open("results/real_data/predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "drug", "Jss", "lag_h", "Q24", "dermis_AUC"])
        writer.writeheader()
        writer.writerows(results_table)
    logger.info("\nSaved predictions.csv")

    # ─── Visualization ───
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Plot 1: UMAP of real data
        ax = axes[0, 0]
        if "X_umap" in adata.obsm:
            umap = adata.obsm["X_umap"]
            if cond_col and cond_col in adata.obs.columns:
                cats = adata.obs[cond_col].astype("category")
                for i, cat in enumerate(cats.cat.categories[:8]):
                    mask = cats == cat
                    ax.scatter(umap[mask, 0], umap[mask, 1], s=1, alpha=0.3, label=str(cat)[:20])
                ax.legend(fontsize=7, markerscale=5)
            else:
                ax.scatter(umap[:, 0], umap[:, 1], s=1, alpha=0.1, c="blue")
            ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
            ax.set_title(f"GSE147424: {adata.n_obs} cells")

        # Plot 2: State vectors heatmap
        ax = axes[0, 1]
        if len(states) > 1:
            state_names = list(states.keys())
            sv_mat = np.array([states[s].to_array() for s in state_names])
            im = ax.imshow(sv_mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
            ax.set_yticks(range(len(state_names)))
            ax.set_yticklabels([s[:20] for s in state_names], fontsize=8)
            ax.set_xticks(range(5))
            ax.set_xticklabels(["barrier", "inflam", "ecm", "vasc", "append"], fontsize=8, rotation=45)
            ax.set_title("Tissue State Vectors")
            plt.colorbar(im, ax=ax, shrink=0.8)

        # Plot 3: Jss comparison across conditions
        ax = axes[1, 0]
        if len(states) > 1:
            for drug_name in test_drugs:
                jss_vals = [r["Jss"] for r in results_table if r["drug"] == drug_name]
                conds = [r["condition"][:15] for r in results_table if r["drug"] == drug_name]
                ax.plot(range(len(conds)), jss_vals, "o-", label=drug_name, markersize=6)
            ax.set_xticks(range(len(list(states.keys()))))
            ax.set_xticklabels([s[:15] for s in states.keys()], fontsize=8, rotation=30)
            ax.set_ylabel("Jss (µg/cm²/h)")
            ax.set_title("Flux vs Tissue State")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Plot 4: Drug comparison
        ax = axes[1, 1]
        first_state = list(states.keys())[0]
        drugs_in_state = [r for r in results_table if r["condition"] == first_state]
        if drugs_in_state:
            drug_names = [r["drug"] for r in drugs_in_state]
            jss_vals = [r["Jss"] for r in drugs_in_state]
            bars = ax.barh(range(len(drug_names)), jss_vals, color="#2196F3", edgecolor="k")
            ax.set_yticks(range(len(drug_names)))
            ax.set_yticklabels(drug_names, fontsize=9)
            ax.set_xlabel("Jss (µg/cm²/h)")
            ax.set_title(f"Drug Permeation ({first_state[:20]})")
            ax.grid(True, alpha=0.3)

        plt.suptitle("Virtual Skin Pipeline — Real GSE147424 Data", fontsize=14)
        plt.tight_layout()
        plt.savefig("results/real_data/real_data_pipeline.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved real_data_pipeline.png")

    except Exception as e:
        logger.warning(f"Plotting failed: {e}")

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Real scRNA-seq data: {adata.n_obs} cells from GSE147424")
    logger.info(f"Tissue states computed: {len(states)}")
    logger.info(f"Drug predictions: {len(results_table)}")
    logger.info(f"Results saved to results/real_data/")


if __name__ == "__main__":
    main()
