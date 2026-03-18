"""
Layer state encoder: encode skin layer-level states from spatial + histology data.

Layer state = {stratum_corneum, viable_epidermis, dermis, appendage_region}
Each layer state vector captures:
  - cell-type composition
  - dominant gene programmes (barrier, differentiation, ECM, …)
  - morphometric parameters (thickness, compactness)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd


# Canonical marker gene sets for skin layer assignment
LAYER_MARKERS: Dict[str, List[str]] = {
    "stratum_corneum": ["FLG", "LOR", "IVL", "SPRR1A", "SPRR2A"],
    "viable_epidermis": ["KRT14", "KRT5", "KRT1", "KRT10", "TP63", "ITGA6"],
    "dermis": ["COL1A1", "COL3A1", "VIM", "DCN", "LUM"],
    "appendage": ["KRT75", "KRT25", "SOX9", "LHX2", "DLX3"],
}


class LayerStateEncoder:
    """Assign spatial spots / cells to skin layers and compute layer-level state vectors."""

    def __init__(self, custom_markers: Optional[Dict[str, List[str]]] = None) -> None:
        self.markers = custom_markers or LAYER_MARKERS

    def assign_layers(self, adata: ad.AnnData) -> ad.AnnData:
        """Score each spot/cell for each layer programme and assign the dominant layer."""
        adata = adata.copy()
        import scanpy as sc

        for layer_name, genes in self.markers.items():
            present = [g for g in genes if g in adata.var_names]
            if len(present) >= 2:
                sc.tl.score_genes(adata, present, score_name=f"score_{layer_name}")
            else:
                adata.obs[f"score_{layer_name}"] = 0.0

        score_cols = [f"score_{k}" for k in self.markers]
        score_mat = adata.obs[score_cols].values
        layer_idx = np.argmax(score_mat, axis=1)
        layer_names = list(self.markers.keys())
        adata.obs["layer_assignment"] = [layer_names[i] for i in layer_idx]
        return adata

    def compute_layer_state(
        self, adata: ad.AnnData, layer_name: str, embedding_key: str = "graphst_emb"
    ) -> np.ndarray:
        """Return average embedding + composition for a single layer."""
        mask = adata.obs["layer_assignment"] == layer_name
        if mask.sum() == 0:
            dim = adata.obsm[embedding_key].shape[1] if embedding_key in adata.obsm else 64
            return np.zeros(dim + len(self.markers))

        emb_mean = adata.obsm[embedding_key][mask].mean(axis=0)
        composition = np.array([
            (adata.obs["layer_assignment"][mask] == k).mean()
            for k in self.markers
        ])
        return np.concatenate([emb_mean, composition])

    def compute_all_layer_states(
        self, adata: ad.AnnData, embedding_key: str = "graphst_emb"
    ) -> Dict[str, np.ndarray]:
        if "layer_assignment" not in adata.obs:
            adata = self.assign_layers(adata)
        return {
            layer: self.compute_layer_state(adata, layer, embedding_key)
            for layer in self.markers
        }

    def layer_composition(self, adata: ad.AnnData) -> pd.DataFrame:
        """Fraction of spots assigned to each layer."""
        if "layer_assignment" not in adata.obs:
            adata = self.assign_layers(adata)
        return adata.obs["layer_assignment"].value_counts(normalize=True).to_frame("fraction")
