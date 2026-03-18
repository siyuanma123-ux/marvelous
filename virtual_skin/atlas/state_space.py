"""
Computable Skin State Space — integrates layer, niche, and cell state encoders
into a unified low-dimensional representation that feeds into the transport model.

Three-tier hierarchy:
  Level 1  layer_state   (SC, VE, dermis, appendage)
  Level 2  niche_state   (perivascular, peri-appendageal, inflammatory, fibrotic, ECM-rich)
  Level 3  cell_state    (per-type sub-states)

The final "tissue state vector" concatenates summary statistics from all three
levels and is the primary interface to the state-modulation layer in the
transport model.
"""

from __future__ import annotations

from typing import Optional

import anndata as ad
import numpy as np

from .layer_state import LayerStateEncoder
from .niche_state import NicheStateEncoder
from .cell_state import CellStateEncoder
from .graphst_wrapper import GraphSTWrapper
from .nicheformer_wrapper import NicheformerWrapper
from .state_vector import TissueStateVector, STATE_AXIS_NAMES


class SkinStateSpace:
    """Orchestrates multi-tier state encoding from raw omics to transport-ready axes."""

    def __init__(
        self,
        graphst: Optional[GraphSTWrapper] = None,
        nicheformer: Optional[NicheformerWrapper] = None,
        layer_enc: Optional[LayerStateEncoder] = None,
        niche_enc: Optional[NicheStateEncoder] = None,
        cell_enc: Optional[CellStateEncoder] = None,
    ) -> None:
        self.graphst = graphst or GraphSTWrapper()
        self.nicheformer = nicheformer
        self.layer_enc = layer_enc or LayerStateEncoder()
        self.niche_enc = niche_enc or NicheStateEncoder()
        self.cell_enc = cell_enc or CellStateEncoder()

    def build_spatial_embedding(self, adata_st: ad.AnnData) -> ad.AnnData:
        """Run GraphST spatial representation learning."""
        return self.graphst.train_representation(adata_st)

    def build_nicheformer_embedding(self, adata_st: ad.AnnData) -> ad.AnnData:
        """Optionally augment with Nicheformer spatial-aware embeddings."""
        if self.nicheformer is not None:
            self.nicheformer.encode_spatial(adata_st)
        return adata_st

    def encode_tissue_state(
        self,
        adata_st: ad.AnnData,
        adata_sc: Optional[ad.AnnData] = None,
    ) -> TissueStateVector:
        """Compute the full tissue state vector from spatial (+ optional scRNA) data.

        Mapping from raw scores to the five canonical state axes:
          barrier_integrity  ← layer(SC barrier_mature) − layer(SC emt_like)
          inflammatory_load  ← niche(inflammatory) + cell(macrophage_m1)
          ecm_remodeling     ← niche(fibrotic) + cell(fibroblast_ecm_remodeling)
          vascularization    ← niche(perivascular) + cell(endothelial mean)
          appendage_openness ← niche(peri_appendageal)
        """
        # Spatial embedding
        if "graphst_emb" not in adata_st.obsm:
            adata_st = self.build_spatial_embedding(adata_st)

        # Layer assignment + scores
        adata_st = self.layer_enc.assign_layers(adata_st)
        layer_states = self.layer_enc.compute_all_layer_states(adata_st)

        # Niche scoring
        adata_st = self.niche_enc.score_niches(adata_st)
        niche_vec = self.niche_enc.niche_state_vector(adata_st)

        # Cell state scoring (on scRNA if available, else on spatial)
        target = adata_sc if adata_sc is not None else adata_st
        target = self.cell_enc.score_all_programs(target)
        cell_vec = self.cell_enc.cell_state_vector(target)

        # Map to canonical axes
        barrier = self._safe_obs_mean(adata_st, "score_stratum_corneum", 0.5)
        emt = self._safe_obs_mean(target, "state_keratinocyte_emt_like", 0.0)
        barrier_integrity = float(np.clip(barrier - emt, 0, 1))

        inflam_niche = self._safe_obs_mean(adata_st, "niche_inflammatory", 0.0)
        m1 = self._safe_obs_mean(target, "state_macrophage_m1_like", 0.0)
        inflammatory_load = float(np.clip((inflam_niche + m1) / 2, 0, 1))

        fibro = self._safe_obs_mean(adata_st, "niche_fibrotic", 0.0)
        ecm_fb = self._safe_obs_mean(target, "state_fibroblast_ecm_remodeling", 0.0)
        ecm_remodeling = float(np.clip((fibro + ecm_fb) / 2, 0, 1))

        vasc_niche = self._safe_obs_mean(adata_st, "niche_perivascular", 0.0)
        endo = self._safe_obs_mean(target, "state_endothelial_arterial", 0.0)
        vascularization = float(np.clip((vasc_niche + endo) / 2, 0, 1))

        append = self._safe_obs_mean(adata_st, "niche_peri_appendageal", 0.0)
        appendage_openness = float(np.clip(append, 0, 1))

        return TissueStateVector(
            barrier_integrity=barrier_integrity,
            inflammatory_load=inflammatory_load,
            ecm_remodeling=ecm_remodeling,
            vascularization=vascularization,
            appendage_openness=appendage_openness,
            layer_state_raw=layer_states,
            niche_state_raw=niche_vec,
            cell_state_raw=cell_vec,
        )

    @staticmethod
    def _safe_obs_mean(adata: ad.AnnData, col: str, default: float) -> float:
        if col in adata.obs.columns:
            return float(adata.obs[col].mean())
        return default
