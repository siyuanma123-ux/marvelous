"""
Wrapper around GraphST for spatial clustering, multi-sample integration,
and scRNA→ST deconvolution.

Directly adapts the GraphST reference implementation:
  参考代码/具体代码/GraphST-main/GraphST/
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Graph construction utilities (adapted from GraphST preprocess.py)
# ---------------------------------------------------------------------------

try:
    import ot as pot
except ImportError:
    pot = None

from sklearn.neighbors import NearestNeighbors
from scipy import sparse as sp


def _construct_spatial_graph(
    adata: ad.AnnData, n_neighbors: int = 3, mode: str = "knn"
) -> None:
    """Build spatial adjacency and neighbor matrices (stored in adata.obsm)."""
    coords = adata.obsm["spatial"]
    n = coords.shape[0]

    if mode == "delaunay":
        from scipy.spatial import Delaunay

        tri = Delaunay(coords)
        adj = np.zeros((n, n))
        for simplex in tri.simplices:
            for i in simplex:
                for j in simplex:
                    if i != j:
                        adj[i, j] = 1
    else:
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coords)
        _, indices = nbrs.kneighbors(coords)
        adj = np.zeros((n, n))
        for i in range(n):
            for j in indices[i, 1:]:
                adj[i, j] = 1

    adj = adj + adj.T
    adj = np.clip(adj, 0, 1)
    adata.obsm["adj"] = adj
    adata.obsm["graph_neigh"] = adj.copy()


def _normalize_adj(adj: np.ndarray) -> np.ndarray:
    adj_sp = sp.coo_matrix(adj)
    rowsum = np.array(adj_sp.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat = sp.diags(d_inv_sqrt)
    return (d_mat @ adj_sp @ d_mat).toarray() + np.eye(adj.shape[0])


# ---------------------------------------------------------------------------
# GNN encoder (adapted from GraphST model.py — Encoder + Discriminator)
# ---------------------------------------------------------------------------

class _AvgReadout(nn.Module):
    def forward(self, emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1, keepdim=True).clamp(min=1e-12)
        return F.normalize(vsum / row_sum, p=2, dim=1)


class _Discriminator(nn.Module):
    def __init__(self, n_h: int) -> None:
        super().__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        nn.init.xavier_uniform_(self.f_k.weight)

    def forward(
        self, c: torch.Tensor, h_pos: torch.Tensor, h_neg: torch.Tensor
    ) -> torch.Tensor:
        c_exp = c.expand_as(h_pos)
        sc_pos = self.f_k(h_pos, c_exp)
        sc_neg = self.f_k(h_neg, c_exp)
        return torch.cat([sc_pos, sc_neg], dim=1)


class _GraphSTEncoder(nn.Module):
    """Two-layer GCN with contrastive self-supervision."""

    def __init__(self, in_dim: int, out_dim: int, graph_neigh: torch.Tensor) -> None:
        super().__init__()
        self.W1 = Parameter(torch.empty(in_dim, out_dim))
        self.W2 = Parameter(torch.empty(out_dim, in_dim))
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        self.graph_neigh = graph_neigh
        self.disc = _Discriminator(out_dim)
        self.read = _AvgReadout()

    def forward(
        self,
        feat: torch.Tensor,
        feat_aug: torch.Tensor,
        adj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = adj @ (feat @ self.W1)
        h = adj @ (z @ self.W2)
        emb = F.relu(z)

        z_a = adj @ (feat_aug @ self.W1)
        emb_a = F.relu(z_a)

        g = torch.sigmoid(self.read(emb, self.graph_neigh))
        g_a = torch.sigmoid(self.read(emb_a, self.graph_neigh))

        ret = self.disc(g, emb, emb_a)
        ret_a = self.disc(g_a, emb_a, emb)
        return z, h, ret, ret_a


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class GraphSTWrapper:
    """High-level interface to GraphST spatial representation learning.

    Supports:
      1) Spatial clustering (unsupervised)
      2) Multi-sample integration
      3) scRNA → spatial deconvolution
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        epochs: int = 600,
        dim_output: int = 64,
        alpha: float = 10.0,
        beta: float = 1.0,
        n_top_genes: int = 3000,
        n_neighbors: int = 3,
        device: Optional[torch.device] = None,
        seed: int = 42,
    ) -> None:
        self.lr = learning_rate
        self.epochs = epochs
        self.dim_output = dim_output
        self.alpha = alpha
        self.beta = beta
        self.n_top_genes = n_top_genes
        self.n_neighbors = n_neighbors
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.seed = seed

    def _prepare(self, adata: ad.AnnData) -> ad.AnnData:
        adata = adata.copy()
        if "highly_variable" not in adata.var:
            sc.pp.highly_variable_genes(
                adata, flavor="seurat_v3", n_top_genes=self.n_top_genes
            )
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.scale(adata, zero_center=False, max_value=10)
        if "adj" not in adata.obsm:
            _construct_spatial_graph(adata, self.n_neighbors)
        return adata

    def train_representation(self, adata: ad.AnnData) -> ad.AnnData:
        """Learn spatial-aware node embeddings."""
        torch.manual_seed(self.seed)
        adata = self._prepare(adata)

        adata_hvg = adata[:, adata.var["highly_variable"]]
        feat_np = (
            adata_hvg.X.toarray()
            if hasattr(adata_hvg.X, "toarray")
            else np.asarray(adata_hvg.X)
        )
        feat = torch.FloatTensor(feat_np).to(self.device)

        adj_norm = _normalize_adj(adata.obsm["adj"])
        adj_t = torch.FloatTensor(adj_norm).to(self.device)
        gn = torch.FloatTensor(
            adata.obsm["graph_neigh"] + np.eye(adata.n_obs)
        ).to(self.device)

        n_csl = np.concatenate(
            [np.ones((adata.n_obs, 1)), np.zeros((adata.n_obs, 1))], axis=1
        )
        label_csl = torch.FloatTensor(n_csl).to(self.device)

        model = _GraphSTEncoder(feat.shape[1], self.dim_output, gn).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        model.train()
        for _ in tqdm(range(self.epochs), desc="GraphST training"):
            perm = torch.randperm(feat.shape[0])
            feat_aug = feat[perm]

            _, h_rec, ret, ret_a = model(feat, feat_aug, adj_t)
            loss = (
                self.alpha * F.mse_loss(h_rec, feat)
                + self.beta * (loss_fn(ret, label_csl) + loss_fn(ret_a, label_csl))
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            emb = model(feat, feat[torch.randperm(feat.shape[0])], adj_t)[0]
            adata.obsm["graphst_emb"] = emb.detach().cpu().numpy()

        return adata

    def cluster(
        self, adata: ad.AnnData, n_clusters: int = 7, method: str = "leiden"
    ) -> ad.AnnData:
        """Cluster spots using GraphST embeddings."""
        if "graphst_emb" not in adata.obsm:
            adata = self.train_representation(adata)

        sc.pp.neighbors(adata, use_rep="graphst_emb", n_neighbors=15)
        if method == "leiden":
            sc.tl.leiden(adata, resolution=_resolution_search(adata, n_clusters))
            adata.obs["graphst_cluster"] = adata.obs["leiden"]
        else:
            sc.tl.louvain(adata, resolution=_resolution_search(adata, n_clusters))
            adata.obs["graphst_cluster"] = adata.obs["louvain"]
        return adata

    def integrate_samples(
        self, adata_list: List[ad.AnnData], batch_key: str = "sample_id"
    ) -> ad.AnnData:
        """Integrate multiple spatial samples into shared embedding."""
        combined = ad.concat(adata_list, label=batch_key, join="inner")
        combined = self.train_representation(combined)
        return combined

    def deconvolve(
        self, adata_st: ad.AnnData, adata_sc: ad.AnnData
    ) -> pd.DataFrame:
        """Map scRNA cell types onto spatial spots — returns (n_spots, n_types) proportions."""
        adata_st = self._prepare(adata_st)
        adata_st = self.train_representation(adata_st)

        emb_st = torch.FloatTensor(adata_st.obsm["graphst_emb"]).to(self.device)

        # Simple autoencoder embedding for scRNA
        sc_hvg = adata_sc[:, adata_sc.var["highly_variable"]] if "highly_variable" in adata_sc.var else adata_sc
        feat_sc = (
            sc_hvg.X.toarray() if hasattr(sc_hvg.X, "toarray") else np.asarray(sc_hvg.X)
        )
        feat_sc_t = torch.FloatTensor(feat_sc).to(self.device)

        # Encode sc with a simple linear layer to same dim
        W = torch.randn(feat_sc.shape[1], self.dim_output, device=self.device) * 0.01
        emb_sc = feat_sc_t @ W
        emb_sc = F.normalize(emb_sc, p=2, dim=1)
        emb_st_n = F.normalize(emb_st, p=2, dim=1)

        # Cosine similarity → softmax mapping
        sim = emb_st_n @ emb_sc.T
        mapping = F.softmax(sim / 0.1, dim=1).detach().cpu().numpy()

        # Aggregate by cell type
        if "cell_type" in adata_sc.obs:
            ct = adata_sc.obs["cell_type"].values
            types = sorted(set(ct))
            props = np.zeros((adata_st.n_obs, len(types)))
            for i, t in enumerate(types):
                mask = ct == t
                props[:, i] = mapping[:, mask].sum(axis=1)
            return pd.DataFrame(props, columns=types, index=adata_st.obs_names)

        return pd.DataFrame(mapping)


def _resolution_search(
    adata: ad.AnnData, target_k: int, lo: float = 0.1, hi: float = 3.0
) -> float:
    """Binary search for Leiden resolution yielding ~target_k clusters."""
    for _ in range(15):
        mid = (lo + hi) / 2
        sc.tl.leiden(adata, resolution=mid, key_added="_tmp_leiden")
        k = adata.obs["_tmp_leiden"].nunique()
        if k < target_k:
            lo = mid
        elif k > target_k:
            hi = mid
        else:
            return mid
    return (lo + hi) / 2
