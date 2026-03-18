"""
COMMOT (COllective optimal transport for cell-cell coMMunication on
spaOTial transcriptomics) wrapper.

Adapted from: 参考代码/具体代码/COMMOT-main/
Uses optimal transport with spatial distance constraints for spatially-resolved CCC.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd


def _kernel_function(
    x: np.ndarray, eta: float, nu: float = 2.0, kernel: str = "exp"
) -> np.ndarray:
    """Spatial decay kernel (from COMMOT reference)."""
    if kernel == "exp":
        return np.exp(-np.power(x / eta, nu))
    elif kernel == "lorentz":
        return 1.0 / (1.0 + np.power(x / eta, nu))
    return np.ones_like(x)


class COMMOTWrapper:
    """Spatially-resolved cell-cell communication via collective optimal transport.

    Uses spatial distance constraints and collective OT to infer
    communication patterns that respect tissue geometry.
    """

    def __init__(
        self,
        dis_thr: float = 500.0,
        kernel: str = "exp",
        eta: float = 200.0,
        nu: float = 2.0,
        cost_type: str = "euc",
    ) -> None:
        self.dis_thr = dis_thr
        self.kernel = kernel
        self.eta = eta
        self.nu = nu
        self.cost_type = cost_type

    def run(
        self,
        adata: ad.AnnData,
        lr_database: Optional[pd.DataFrame] = None,
    ) -> ad.AnnData:
        """Run COMMOT spatial CCC analysis.

        Requires adata.obsm['spatial'] for coordinates and the commot package.
        Falls back to a simplified implementation if commot is not installed.
        """
        try:
            import commot as ct

            ct.tl.spatial_communication(
                adata,
                database_name="CellChat",
                species="human",
                dis_thr=self.dis_thr,
                heteromeric=True,
            )
            return adata
        except ImportError:
            return self._fallback_spatial_ccc(adata, lr_database)

    def _fallback_spatial_ccc(
        self, adata: ad.AnnData, lr_db: Optional[pd.DataFrame] = None
    ) -> ad.AnnData:
        """Simplified spatial CCC when commot is not installed.

        Computes spatial co-expression scores weighted by distance kernel.
        """
        coords = adata.obsm["spatial"]
        from scipy.spatial import distance_matrix

        dmat = distance_matrix(coords, coords)
        W = _kernel_function(dmat, self.eta, self.nu, self.kernel)
        W[dmat > self.dis_thr] = 0.0
        np.fill_diagonal(W, 0.0)

        if lr_db is None:
            lr_db = self._default_skin_lr_pairs()

        X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        gene_names = list(adata.var_names)

        scores = {}
        for _, row in lr_db.iterrows():
            lig, rec = row["ligand"], row["receptor"]
            if lig not in gene_names or rec not in gene_names:
                continue
            li_idx = gene_names.index(lig)
            ri_idx = gene_names.index(rec)

            lig_expr = X[:, li_idx]
            rec_expr = X[:, ri_idx]

            # Spatial co-expression: weighted outer product sum
            score_per_spot = W @ (lig_expr[:, None] * rec_expr[None, :]).sum(axis=0)
            scores[f"{lig}_{rec}"] = score_per_spot

        if scores:
            score_df = pd.DataFrame(scores, index=adata.obs_names)
            adata.obsm["commot_scores"] = score_df.values
            adata.uns["commot_lr_names"] = list(scores.keys())

        return adata

    def get_communication_vectors(self, adata: ad.AnnData) -> np.ndarray:
        """Extract per-spot communication score matrix."""
        if "commot_scores" in adata.obsm:
            return adata.obsm["commot_scores"]
        return np.zeros((adata.n_obs, 1))

    @staticmethod
    def _default_skin_lr_pairs() -> pd.DataFrame:
        """Minimal set of skin-relevant ligand-receptor pairs."""
        pairs = [
            ("TGFB1", "TGFBR1"), ("TGFB1", "TGFBR2"),
            ("TNF", "TNFRSF1A"), ("IL1B", "IL1R1"),
            ("IL6", "IL6R"), ("VEGFA", "KDR"),
            ("VEGFA", "FLT1"), ("PDGFB", "PDGFRB"),
            ("WNT5A", "FZD1"), ("BMP2", "BMPR1A"),
            ("FGF2", "FGFR1"), ("EGF", "EGFR"),
            ("CCL2", "CCR2"), ("CXCL12", "CXCR4"),
            ("DLL4", "NOTCH1"), ("JAG1", "NOTCH2"),
            ("COL1A1", "ITGA1"), ("FN1", "ITGA5"),
            ("SPP1", "CD44"), ("SHH", "PTCH1"),
        ]
        return pd.DataFrame(pairs, columns=["ligand", "receptor"])
