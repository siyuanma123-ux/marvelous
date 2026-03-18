"""
Consensus crosstalk module builder.

Integrates CellChat (pathway-level), COMMOT (spatial OT), and FlowSig
(directional flow) into unified consensus modules that can directly
modulate transport parameters.

Five consensus modules:
  1. Barrier Maintenance Module
  2. Inflammatory Permeability Module
  3. ECM Retention / Sink Module
  4. Vascular Clearance Module
  5. Appendage Bypass Module
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd

from .cellchat_wrapper import CellChatWrapper
from .commot_wrapper import COMMOTWrapper
from .flowsig_module import FlowSigModule


@dataclass
class CrosstalkModule:
    """A consensus crosstalk module with quantified activity."""

    name: str
    description: str

    # Activity score [0, 1] — higher means stronger modulation
    activity: float = 0.0

    # Contributing pathways (from CellChat)
    cellchat_pathways: List[str] = field(default_factory=list)
    cellchat_strength: float = 0.0

    # Spatial communication patterns (from COMMOT)
    commot_strength: float = 0.0

    # Directional flows (from FlowSig)
    flowsig_flows: List[Dict] = field(default_factory=list)
    flowsig_strength: float = 0.0

    # Confidence: agreement across methods
    consensus_confidence: float = 0.0

    def compute_consensus_activity(self) -> float:
        """Weighted average of method-specific strengths."""
        weights = np.array([0.4, 0.3, 0.3])
        strengths = np.array([
            self.cellchat_strength,
            self.commot_strength,
            self.flowsig_strength,
        ])
        strengths = np.clip(strengths, 0, 1)
        self.activity = float(np.dot(weights, strengths))

        # Confidence: how well do methods agree?
        if strengths.std() < 0.1:
            self.consensus_confidence = 1.0
        elif strengths.std() < 0.25:
            self.consensus_confidence = 0.7
        else:
            self.consensus_confidence = 0.4

        return self.activity


SKIN_MODULE_DEFINITIONS = {
    "barrier_maintenance": {
        "description": "Keratinocyte differentiation, tight junctions, barrier homeostasis",
        "pathway_keywords": ["WNT", "NOTCH", "BMP", "CLDN", "DSG", "KRT"],
        "target_param": "D_sc",
        "direction": "negative",  # high barrier → low D_sc
    },
    "inflammatory_permeability": {
        "description": "Inflammation-driven permeability increase",
        "pathway_keywords": ["TNF", "IL1", "IL6", "CXCL", "CCL", "IFN", "S100A"],
        "target_param": "D_sc",
        "direction": "positive",  # high inflammation → high D_sc
    },
    "ecm_retention": {
        "description": "ECM remodeling and dermal drug retention/binding",
        "pathway_keywords": ["TGFB", "COL", "FN1", "MMP", "LOX", "POSTN", "FAP"],
        "target_param": "k_bind_dermis",
        "direction": "positive",
    },
    "vascular_clearance": {
        "description": "Vascular niche and immune–endothelial crosstalk → dermal clearance",
        "pathway_keywords": ["VEGF", "PDGF", "ANGPT", "DLL", "EPH"],
        "target_param": "k_clear_vasc",
        "direction": "positive",
    },
    "appendage_bypass": {
        "description": "Follicular and glandular bypass pathway",
        "pathway_keywords": ["SHH", "WNT", "BMP", "FGF", "EDA"],
        "target_param": "w_appendage",
        "direction": "positive",
    },
}


class ConsensusCrosstalkBuilder:
    """Build consensus modules from multiple CCC methods."""

    def __init__(
        self,
        cellchat: Optional[CellChatWrapper] = None,
        commot: Optional[COMMOTWrapper] = None,
        flowsig: Optional[FlowSigModule] = None,
        module_defs: Optional[Dict[str, Dict]] = None,
    ) -> None:
        self.cellchat = cellchat or CellChatWrapper()
        self.commot = commot or COMMOTWrapper()
        self.flowsig = flowsig or FlowSigModule()
        self.defs = module_defs or SKIN_MODULE_DEFINITIONS
        self.modules: Dict[str, CrosstalkModule] = {}

    def build(
        self,
        adata: ad.AnnData,
        groupby: str = "cell_type",
    ) -> Dict[str, CrosstalkModule]:
        """Run all CCC methods and build consensus modules."""
        # CellChat-style analysis
        cc_results = self.cellchat.run(adata, groupby=groupby)
        cc_categorized = self.cellchat.identify_skin_relevant_pathways(cc_results)

        # COMMOT spatial analysis
        adata = self.commot.run(adata)
        commot_vec = self.commot.get_communication_vectors(adata)

        # FlowSig directional flow
        self.flowsig.fit(adata)
        fs_categorized = self.flowsig.identify_transport_relevant_flows()

        # Build each consensus module
        for mod_name, mod_def in self.defs.items():
            module = CrosstalkModule(
                name=mod_name,
                description=mod_def["description"],
            )

            # CellChat contribution — map module names to CellChat category keys
            cc_key_map = {
                "barrier_maintenance": "barrier",
                "inflammatory_permeability": "inflammatory",
                "ecm_retention": "ecm_remodeling",
                "vascular_clearance": "vascular",
                "appendage_bypass": "appendage",
            }
            cc_key = cc_key_map.get(mod_name, mod_name)
            if cc_key in cc_categorized:
                df = cc_categorized[cc_key]
                module.cellchat_pathways = df["ligand_complex"].unique().tolist()[:10]
                module.cellchat_strength = min(1.0, len(df) / 50.0)

            # COMMOT contribution
            if "commot_lr_names" in adata.uns:
                lr_names = adata.uns["commot_lr_names"]
                keywords = mod_def["pathway_keywords"]
                matching = [
                    n for n in lr_names
                    if any(kw.lower() in n.lower() for kw in keywords)
                ]
                if matching and commot_vec.shape[1] > 0:
                    indices = [lr_names.index(m) for m in matching if m in lr_names]
                    if indices:
                        module.commot_strength = float(
                            np.clip(commot_vec[:, indices].mean(), 0, 1)
                        )

            # FlowSig contribution
            fs_key_map = {
                "barrier_maintenance": "barrier_maintenance",
                "inflammatory_permeability": "inflammatory_permeability",
                "ecm_retention": "ecm_retention",
                "vascular_clearance": "vascular_clearance",
                "appendage_bypass": "appendage_bypass",
            }
            fs_key = fs_key_map.get(mod_name)
            if fs_key and fs_key in fs_categorized:
                df = fs_categorized[fs_key]
                module.flowsig_flows = df.head(10).to_dict("records")
                module.flowsig_strength = float(np.clip(df["strength"].abs().mean(), 0, 1))

            module.compute_consensus_activity()
            self.modules[mod_name] = module

        return self.modules

    def module_activity_vector(self) -> np.ndarray:
        """Return (n_modules,) activity vector aligned with state-axis order."""
        return np.array(
            [self.modules.get(m, CrosstalkModule(m, "")).activity for m in self.defs],
            dtype=np.float32,
        )

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, mod in self.modules.items():
            rows.append({
                "module": name,
                "activity": mod.activity,
                "confidence": mod.consensus_confidence,
                "cellchat_strength": mod.cellchat_strength,
                "commot_strength": mod.commot_strength,
                "flowsig_strength": mod.flowsig_strength,
                "target_param": self.defs[name]["target_param"],
                "direction": self.defs[name]["direction"],
            })
        return pd.DataFrame(rows)
