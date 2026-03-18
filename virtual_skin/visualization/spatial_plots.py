"""Spatial transcriptomics and state visualisation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import anndata as ad
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
except ImportError:
    plt = None


def plot_spatial_state(
    adata: ad.AnnData,
    color_by: str = "layer_assignment",
    coords_key: str = "spatial",
    figsize: tuple = (8, 8),
    title: Optional[str] = None,
    save: Optional[str] = None,
) -> None:
    """Scatter plot of spatial spots coloured by state assignment."""
    if plt is None:
        raise ImportError("matplotlib required for plotting")

    coords = adata.obsm[coords_key]
    labels = adata.obs[color_by]
    categories = sorted(labels.unique())

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.get_cmap("tab10", len(categories))

    for i, cat in enumerate(categories):
        mask = labels == cat
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=8, c=[cmap(i)], label=cat, alpha=0.7,
        )

    ax.set_xlabel("Spatial X (µm)")
    ax.set_ylabel("Spatial Y (µm)")
    ax.set_title(title or f"Spatial distribution: {color_by}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_aspect("equal")
    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    plt.show()


def plot_niche_map(
    adata: ad.AnnData,
    niche_scores: Optional[List[str]] = None,
    coords_key: str = "spatial",
    figsize: tuple = (16, 4),
    save: Optional[str] = None,
) -> None:
    """Heatmap-style spatial plots for each niche score."""
    if plt is None:
        raise ImportError("matplotlib required for plotting")

    if niche_scores is None:
        niche_scores = [c for c in adata.obs.columns if c.startswith("niche_")]

    n = len(niche_scores)
    if n == 0:
        return

    coords = adata.obsm[coords_key]
    fig, axes = plt.subplots(1, n, figsize=(figsize[0], figsize[1]))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, niche_scores):
        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=adata.obs[col].values, s=6, cmap="RdYlBu_r", alpha=0.8,
        )
        ax.set_title(col.replace("niche_", ""), fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(sc, ax=ax, shrink=0.7)

    plt.suptitle("Spatial niche scores", fontsize=12)
    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    plt.show()
