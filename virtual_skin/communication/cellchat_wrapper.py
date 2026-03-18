"""
CellChat-style cell-cell communication inference.

Integrates the CellChat probability model via the liana-py framework
(参考代码/具体代码/liana-py-main/), which provides a unified Pythonic
interface to CellChat, CellPhoneDB, NATMI, and other methods.

For native R CellChat usage, see:
  参考代码/具体代码/CellChat-main/R/modeling.R
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd


class CellChatWrapper:
    """Python-side CellChat communication analysis via liana.

    Computes pathway-level communication probabilities between cell groups,
    identifying which signalling pathways dominate in each spatial niche.
    """

    def __init__(
        self,
        resource_name: str = "CellChatDB",
        species: str = "human",
        min_cells: int = 10,
        p_threshold: float = 0.05,
        n_perms: int = 1000,
    ) -> None:
        self.resource_name = resource_name
        self.species = species
        self.min_cells = min_cells
        self.p_threshold = p_threshold
        self.n_perms = n_perms

    def run(
        self,
        adata: ad.AnnData,
        groupby: str = "cell_type",
        use_raw: bool = False,
    ) -> pd.DataFrame:
        """Run CellChat-like analysis via liana.

        Returns DataFrame with columns:
          source, target, ligand_complex, receptor_complex,
          lr_probs, cellchat_pvals, pathway_name
        """
        try:
            import liana as li
        except ImportError:
            raise ImportError("liana is required: pip install liana")

        li.mt.cellphonedb(
            adata,
            groupby=groupby,
            resource_name=self.resource_name,
            use_raw=use_raw,
            n_perms=self.n_perms,
            verbose=True,
            key_added="liana_res",
        )

        results = adata.uns["liana_res"]
        if self.p_threshold < 1.0:
            if "cellphone_pvals" in results.columns:
                results = results[results["cellphone_pvals"] < self.p_threshold]

        return results

    def run_cellchat_native(
        self,
        adata: ad.AnnData,
        groupby: str = "cell_type",
    ) -> pd.DataFrame:
        """Direct CellChat-style scoring using liana's CellChat method."""
        try:
            import liana as li
        except ImportError:
            raise ImportError("liana is required: pip install liana")

        li.mt.rank_aggregate(
            adata,
            groupby=groupby,
            resource_name=self.resource_name,
            use_raw=False,
            verbose=True,
            key_added="cellchat_res",
        )
        return adata.uns["cellchat_res"]

    def pathway_summary(self, results: pd.DataFrame) -> pd.DataFrame:
        """Aggregate communication strengths per pathway."""
        if "ligand_complex" not in results.columns:
            return pd.DataFrame()

        # Aggregate by pathway or ligand-receptor pair
        group_cols = []
        if "pathway_name" in results.columns:
            group_cols = ["pathway_name"]
        else:
            group_cols = ["ligand_complex", "receptor_complex"]

        agg = (
            results.groupby(group_cols)
            .agg(
                n_interactions=("source", "count"),
                mean_magnitude=("magnitude_rank", "mean") if "magnitude_rank" in results.columns else ("source", "count"),
            )
            .sort_values("n_interactions", ascending=False)
            .reset_index()
        )
        return agg

    def source_target_network(
        self, results: pd.DataFrame
    ) -> Dict[Tuple[str, str], float]:
        """Build directed weighted network: (source, target) → aggregate strength."""
        network = {}
        for _, row in results.iterrows():
            key = (row["source"], row["target"])
            score = row.get("lr_probs", row.get("magnitude_rank", 1.0))
            if isinstance(score, (int, float)) and not np.isnan(score):
                network[key] = network.get(key, 0) + score
        return network

    def identify_skin_relevant_pathways(
        self, results: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Filter pathways relevant to transport modulation."""
        relevant = {
            "barrier": ["WNT", "NOTCH", "BMP", "CLDN", "DSG", "DSC"],
            "inflammatory": ["TNF", "IL1", "IL6", "CXCL", "CCL", "IFN"],
            "ecm_remodeling": ["TGF", "FN1", "COL", "MMP", "TIMP", "LOX"],
            "vascular": ["VEGF", "PDGF", "ANGPT", "EPH", "DLL"],
            "appendage": ["WNT", "SHH", "BMP", "FGF", "EDA"],
        }

        categorized = {}
        for module_name, keywords in relevant.items():
            mask = results.apply(
                lambda row: any(
                    kw.lower() in str(row.get("ligand_complex", "")).lower()
                    or kw.lower() in str(row.get("receptor_complex", "")).lower()
                    for kw in keywords
                ),
                axis=1,
            )
            categorized[module_name] = results[mask].copy()
        return categorized
