"""
FlowSig-style directional intercellular signalling flow module.

FlowSig organises communication as:
  inflow signals → intracellular gene expression modules (GEMs) → outflow signals

This creates a directed information-flow graph rather than static L-R edges,
enabling identification of how external signals are processed intracellularly
and converted to downstream secreted outputs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


class FlowSigModule:
    """Directional intercellular signalling flow analysis.

    Implements a simplified FlowSig-like model:
      1. Define inflow signals (ligands received)
      2. Identify intracellular gene expression modules (GEMs) via NMF
      3. Define outflow signals (ligands secreted)
      4. Learn directed flow: inflow → GEM → outflow via conditional Granger-like tests
    """

    def __init__(
        self,
        n_gems: int = 5,
        n_inflow: int = 20,
        n_outflow: int = 20,
    ) -> None:
        self.n_gems = n_gems
        self.n_inflow = n_inflow
        self.n_outflow = n_outflow

        self.gem_components: Optional[np.ndarray] = None
        self.flow_graph: Optional[Dict[str, Any]] = None

    def fit(
        self,
        adata: ad.AnnData,
        inflow_genes: Optional[List[str]] = None,
        outflow_genes: Optional[List[str]] = None,
        receptor_genes: Optional[List[str]] = None,
        ligand_genes: Optional[List[str]] = None,
    ) -> "FlowSigModule":
        """Fit the directional flow model.

        Steps:
          1. Extract inflow signal matrix (receptor expression)
          2. Decompose intracellular expression into GEMs via NMF
          3. Extract outflow signal matrix (ligand expression)
          4. Compute directed associations: inflow→GEM, GEM→outflow
        """
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        gene_names = list(adata.var_names)

        # Define inflow (receptors) and outflow (ligands)
        if inflow_genes is None:
            inflow_genes = receptor_genes or self._default_receptor_genes()
        if outflow_genes is None:
            outflow_genes = ligand_genes or self._default_ligand_genes()

        inflow_idx = [gene_names.index(g) for g in inflow_genes if g in gene_names]
        outflow_idx = [gene_names.index(g) for g in outflow_genes if g in gene_names]

        inflow_mat = X[:, inflow_idx]   # (n_cells, n_receptors)
        outflow_mat = X[:, outflow_idx]  # (n_cells, n_ligands)

        # NMF on full expression → GEMs
        from sklearn.decomposition import NMF

        X_pos = np.clip(X, 0, None)
        nmf = NMF(n_components=self.n_gems, max_iter=500, random_state=42)
        gem_scores = nmf.fit_transform(X_pos)  # (n_cells, n_gems)
        self.gem_components = nmf.components_  # (n_gems, n_genes)

        # Directed association: inflow → GEM
        inflow_to_gem = np.zeros((len(inflow_idx), self.n_gems))
        for i in range(len(inflow_idx)):
            for g in range(self.n_gems):
                r, _ = pearsonr(inflow_mat[:, i], gem_scores[:, g])
                inflow_to_gem[i, g] = r if not np.isnan(r) else 0.0

        # Directed association: GEM → outflow
        gem_to_outflow = np.zeros((self.n_gems, len(outflow_idx)))
        for g in range(self.n_gems):
            for j in range(len(outflow_idx)):
                r, _ = pearsonr(gem_scores[:, g], outflow_mat[:, j])
                gem_to_outflow[g, j] = r if not np.isnan(r) else 0.0

        inflow_names = [inflow_genes[i] for i in range(len(inflow_idx)) if i < len(inflow_genes)]
        outflow_names = [outflow_genes[j] for j in range(len(outflow_idx)) if j < len(outflow_genes)]

        self.flow_graph = {
            "inflow_genes": inflow_names[:len(inflow_idx)],
            "outflow_genes": outflow_names[:len(outflow_idx)],
            "n_gems": self.n_gems,
            "inflow_to_gem": inflow_to_gem,
            "gem_to_outflow": gem_to_outflow,
            "gem_scores": gem_scores,
        }

        # Store in adata
        adata.obsm["flowsig_gem_scores"] = gem_scores
        adata.uns["flowsig_graph"] = {
            "inflow_to_gem": inflow_to_gem,
            "gem_to_outflow": gem_to_outflow,
        }

        return self

    def get_flow_strength(
        self, module_type: str = "all"
    ) -> pd.DataFrame:
        """Get flow-through strength for each inflow→GEM→outflow path."""
        if self.flow_graph is None:
            raise RuntimeError("Call fit() first.")

        in2gem = self.flow_graph["inflow_to_gem"]
        gem2out = self.flow_graph["gem_to_outflow"]
        in_names = self.flow_graph["inflow_genes"]
        out_names = self.flow_graph["outflow_genes"]

        # Full path strength = in→gem × gem→out
        rows = []
        for i, ig in enumerate(in_names):
            for g in range(self.n_gems):
                for j, og in enumerate(out_names):
                    strength = in2gem[i, g] * gem2out[g, j]
                    if abs(strength) > 0.01:
                        rows.append({
                            "inflow": ig,
                            "gem": g,
                            "outflow": og,
                            "strength": strength,
                        })

        return pd.DataFrame(rows).sort_values("strength", ascending=False, key=abs)

    def identify_transport_relevant_flows(self) -> Dict[str, pd.DataFrame]:
        """Categorise flows into transport-relevant modules."""
        df = self.get_flow_strength()
        if df.empty:
            return {}

        categories = {
            "barrier_maintenance": ["NOTCH", "WNT", "BMP", "CLDN", "KRT"],
            "inflammatory_permeability": ["TNF", "IL1", "IL6", "CXCL", "CCL", "IFN"],
            "ecm_retention": ["TGFB", "COL", "FN1", "MMP", "LOX", "POSTN"],
            "vascular_clearance": ["VEGF", "PDGF", "ANGPT", "DLL", "NOTCH"],
            "appendage_bypass": ["SHH", "WNT", "BMP", "FGF", "EDA"],
        }

        result = {}
        for cat, keywords in categories.items():
            mask = df.apply(
                lambda row: any(
                    kw.lower() in row["inflow"].lower() or kw.lower() in row["outflow"].lower()
                    for kw in keywords
                ),
                axis=1,
            )
            if mask.any():
                result[cat] = df[mask].copy()

        return result

    @staticmethod
    def _default_receptor_genes() -> List[str]:
        return [
            "TGFBR1", "TGFBR2", "TNFRSF1A", "IL1R1", "IL6R",
            "KDR", "FLT1", "PDGFRB", "FZD1", "BMPR1A",
            "FGFR1", "EGFR", "CCR2", "CXCR4", "NOTCH1",
            "NOTCH2", "ITGA1", "ITGA5", "CD44", "PTCH1",
        ]

    @staticmethod
    def _default_ligand_genes() -> List[str]:
        return [
            "TGFB1", "TGFB2", "TNF", "IL1B", "IL6",
            "VEGFA", "PDGFB", "WNT5A", "BMP2", "FGF2",
            "EGF", "CCL2", "CXCL12", "DLL4", "JAG1",
            "COL1A1", "FN1", "SPP1", "SHH", "IL10",
        ]
