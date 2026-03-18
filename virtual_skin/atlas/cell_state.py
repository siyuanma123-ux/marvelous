"""
Cell state encoder: identify cell-type sub-states relevant to transport modulation.

Key cell types and their transport-relevant states:
  - Keratinocytes: basal stem, differentiating, EMT-like, barrier-mature
  - Fibroblasts: resting, inflammatory, ECM-remodeling, fibrotic
  - Endothelial: arterial, venous, lymphatic, tip cell
  - Macrophages: M1-like, M2-like, tissue-resident, perivascular
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


CELL_STATE_PROGRAMS: Dict[str, Dict[str, List[str]]] = {
    "keratinocyte": {
        "basal_stem": ["KRT14", "KRT5", "TP63", "ITGA6", "COL17A1"],
        "differentiating": ["KRT1", "KRT10", "DSP", "DSG1"],
        "barrier_mature": ["FLG", "LOR", "IVL", "CDSN"],
        "emt_like": ["VIM", "SNAI2", "ZEB1", "CDH2", "FN1"],
    },
    "fibroblast": {
        "resting": ["DCN", "LUM", "FBLN1", "GSN"],
        "inflammatory": ["IL6", "CXCL1", "CXCL8", "CCL2", "PTGS2"],
        "ecm_remodeling": ["COL1A1", "COL3A1", "FN1", "LOX", "MMP2"],
        "fibrotic": ["ACTA2", "POSTN", "FAP", "TGFB1", "CTGF"],
    },
    "endothelial": {
        "arterial": ["GJA5", "HEY1", "EFNB2", "DLL4"],
        "venous": ["ACKR1", "NR2F2", "SELP"],
        "lymphatic": ["PROX1", "LYVE1", "FLT4", "PDPN"],
        "tip_cell": ["ESM1", "PGF", "APLN", "KDR"],
    },
    "macrophage": {
        "m1_like": ["TNF", "IL1B", "NOS2", "CD80", "CXCL10"],
        "m2_like": ["MRC1", "CD163", "IL10", "TGFB1", "ARG1"],
        "tissue_resident": ["C1QA", "C1QB", "C1QC", "MERTK", "CD68"],
        "perivascular": ["LYVE1", "MRC1", "STAB1", "F13A1"],
    },
}


class CellStateEncoder:
    """Score and classify cell sub-states for each major skin cell type."""

    def __init__(
        self, custom_programs: Optional[Dict[str, Dict[str, List[str]]]] = None
    ) -> None:
        self.programs = custom_programs or CELL_STATE_PROGRAMS

    def score_all_programs(self, adata: ad.AnnData) -> ad.AnnData:
        adata = adata.copy()
        for cell_type, states in self.programs.items():
            for state_name, genes in states.items():
                col = f"state_{cell_type}_{state_name}"
                present = [g for g in genes if g in adata.var_names]
                if len(present) >= 2:
                    sc.tl.score_genes(adata, present, score_name=col)
                else:
                    adata.obs[col] = 0.0
        return adata

    def assign_dominant_state(
        self, adata: ad.AnnData, cell_type: str
    ) -> pd.Series:
        """For a given cell type, return the dominant sub-state per cell."""
        states = self.programs.get(cell_type, {})
        cols = [f"state_{cell_type}_{s}" for s in states]
        existing = [c for c in cols if c in adata.obs.columns]
        if not existing:
            return pd.Series("unknown", index=adata.obs_names)
        scores = adata.obs[existing].values
        idx = np.argmax(scores, axis=1)
        names = [c.replace(f"state_{cell_type}_", "") for c in existing]
        return pd.Series([names[i] for i in idx], index=adata.obs_names)

    def cell_state_vector(
        self, adata: ad.AnnData, cell_type: Optional[str] = None
    ) -> np.ndarray:
        """Sample-level mean state scores for specified or all cell types."""
        if cell_type:
            types = {cell_type: self.programs[cell_type]}
        else:
            types = self.programs
        vals = []
        for ct, states in types.items():
            for sn in states:
                col = f"state_{ct}_{sn}"
                if col in adata.obs.columns:
                    vals.append(adata.obs[col].mean())
                else:
                    vals.append(0.0)
        return np.array(vals, dtype=np.float32)

    def cell_state_matrix(self, adata: ad.AnnData) -> Tuple[np.ndarray, List[str]]:
        """Per-cell state score matrix (n_cells, n_states) and column names."""
        cols = []
        for ct, states in self.programs.items():
            for sn in states:
                cols.append(f"state_{ct}_{sn}")
        existing = [c for c in cols if c in adata.obs.columns]
        return adata.obs[existing].values.astype(np.float32), existing
