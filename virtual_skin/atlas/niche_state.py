"""
Niche state encoder: identify and quantify spatial micro-environment niches.

Niche categories relevant to transdermal transport:
  - perivascular niche
  - peri-appendageal niche
  - inflammatory niche
  - fibrotic niche
  - ECM-rich dermal niche
"""

from __future__ import annotations

from typing import Dict, List, Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


# Gene-programme signatures for each niche type
NICHE_SIGNATURES: Dict[str, List[str]] = {
    "perivascular": [
        "PECAM1", "CDH5", "VWF", "KDR", "CLDN5", "FLT1",
        "PDGFRB", "ACTA2", "RGS5", "MCAM",
    ],
    "peri_appendageal": [
        "KRT75", "KRT25", "SOX9", "LHX2", "DLX3", "WNT5A",
        "BMP2", "SHH", "LEF1",
    ],
    "inflammatory": [
        "IL1B", "TNF", "IL6", "CXCL8", "CCL2", "PTGS2",
        "S100A8", "S100A9", "NFKBIA",
    ],
    "fibrotic": [
        "COL1A1", "COL3A1", "FN1", "ACTA2", "TGFB1", "TGFB2",
        "CTGF", "LOX", "POSTN", "FAP",
    ],
    "ecm_rich_dermal": [
        "DCN", "LUM", "BGN", "VCAN", "COL6A1", "COL14A1",
        "FBLN1", "ELN", "FBN1",
    ],
}


class NicheStateEncoder:
    """Score spatial niches and extract low-dimensional niche state axes."""

    def __init__(
        self, custom_signatures: Optional[Dict[str, List[str]]] = None
    ) -> None:
        self.signatures = custom_signatures or NICHE_SIGNATURES

    def score_niches(self, adata: ad.AnnData) -> ad.AnnData:
        """Add niche scores to adata.obs for each micro-environment signature."""
        adata = adata.copy()
        for niche_name, genes in self.signatures.items():
            present = [g for g in genes if g in adata.var_names]
            if len(present) >= 3:
                sc.tl.score_genes(adata, present, score_name=f"niche_{niche_name}")
            else:
                adata.obs[f"niche_{niche_name}"] = 0.0
        return adata

    def assign_dominant_niche(self, adata: ad.AnnData) -> ad.AnnData:
        if not any(c.startswith("niche_") for c in adata.obs.columns):
            adata = self.score_niches(adata)
        niche_cols = [f"niche_{k}" for k in self.signatures]
        scores = adata.obs[niche_cols].values
        idx = np.argmax(scores, axis=1)
        names = list(self.signatures.keys())
        adata.obs["dominant_niche"] = [names[i] for i in idx]
        return adata

    def niche_state_vector(self, adata: ad.AnnData) -> np.ndarray:
        """Global niche state: mean score per niche across all spots → low-dim vector."""
        if not any(c.startswith("niche_") for c in adata.obs.columns):
            adata = self.score_niches(adata)
        cols = [f"niche_{k}" for k in self.signatures]
        return adata.obs[cols].mean(axis=0).values.astype(np.float32)

    def spot_niche_matrix(self, adata: ad.AnnData) -> np.ndarray:
        """Per-spot niche score matrix (n_spots, n_niches)."""
        if not any(c.startswith("niche_") for c in adata.obs.columns):
            adata = self.score_niches(adata)
        cols = [f"niche_{k}" for k in self.signatures]
        return adata.obs[cols].values.astype(np.float32)

    def niche_enrichment_by_layer(
        self, adata: ad.AnnData, layer_col: str = "layer_assignment"
    ) -> pd.DataFrame:
        """Mean niche scores stratified by skin layer."""
        if not any(c.startswith("niche_") for c in adata.obs.columns):
            adata = self.score_niches(adata)
        cols = [f"niche_{k}" for k in self.signatures]
        return adata.obs.groupby(layer_col)[cols].mean()
