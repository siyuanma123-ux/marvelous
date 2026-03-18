"""
Unified loader for single-cell and spatial transcriptomics data,
with quality control, batch correction, and state alignment pipelines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import scanpy as sc


def load_scrna(
    path: str | Path,
    min_genes: int = 200,
    min_cells: int = 3,
    max_mt_pct: float = 20.0,
    n_top_genes: int = 3000,
    batch_key: Optional[str] = None,
) -> ad.AnnData:
    """Load, QC, and preprocess scRNA-seq data.

    Follows standard Scanpy pipeline consistent with HSCA and
    skin_fibroblast_atlas reference workflows.
    """
    adata = sc.read(str(path))

    # Basic QC
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    adata = adata[adata.obs["pct_counts_mt"] < max_mt_pct].copy()

    # Normalization
    adata.layers["raw_counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["log_norm"] = adata.X.copy()

    # HVG selection
    sc.pp.highly_variable_genes(
        adata, n_top_genes=n_top_genes, flavor="seurat_v3", layer="raw_counts"
    )

    # PCA + neighbors + UMAP
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=50, use_highly_variable=True)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata)

    # Optional batch correction via Harmony
    if batch_key and batch_key in adata.obs.columns:
        try:
            import scanpy.external as sce
            sce.pp.harmony_integrate(adata, batch_key, adjusted_basis="X_pca_harmony")
            sc.pp.neighbors(adata, use_rep="X_pca_harmony")
            sc.tl.umap(adata)
        except ImportError:
            import warnings
            warnings.warn("harmonypy not installed — skipping batch correction. Install via: pip install harmonypy")

    return adata


def load_spatial(
    path: str | Path,
    library_id: Optional[str] = None,
    n_top_genes: int = 3000,
) -> ad.AnnData:
    """Load and preprocess spatial transcriptomics data (10X Visium / Stereo-seq)."""
    path = Path(path)
    if path.suffix == ".h5ad":
        adata = sc.read_h5ad(str(path))
    else:
        adata = sc.read_visium(str(path), library_id=library_id)

    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_cells=5)

    adata.layers["raw_counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(
        adata, n_top_genes=n_top_genes, flavor="seurat_v3", layer="raw_counts"
    )

    return adata


def load_multimodal_paired(
    sc_path: str | Path,
    st_path: str | Path,
    **kwargs: Any,
) -> Tuple[ad.AnnData, ad.AnnData]:
    """Load matched scRNA-seq + spatial data from the same donor/site."""
    adata_sc = load_scrna(sc_path, **kwargs)
    adata_st = load_spatial(st_path, **kwargs)

    overlap_genes = sorted(
        set(adata_sc.var_names) & set(adata_st.var_names)
    )
    adata_sc.uns["overlap_genes"] = overlap_genes
    adata_st.uns["overlap_genes"] = overlap_genes

    return adata_sc, adata_st


def align_gene_space(
    adata_sc: ad.AnnData, adata_st: ad.AnnData
) -> Tuple[ad.AnnData, ad.AnnData]:
    """Restrict both AnnData objects to shared HVG space."""
    hvg_sc = set(adata_sc.var_names[adata_sc.var["highly_variable"]])
    hvg_st = set(adata_st.var_names[adata_st.var["highly_variable"]])
    shared = sorted(hvg_sc & hvg_st)
    if len(shared) < 500:
        shared = sorted(set(adata_sc.var_names) & set(adata_st.var_names))

    return adata_sc[:, shared].copy(), adata_st[:, shared].copy()
