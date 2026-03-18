"""Validation result visualisation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from scipy import stats
except ImportError:
    plt = None


def plot_observed_vs_predicted(
    df: pd.DataFrame,
    obs_col: str = "obs_Jss",
    pred_col: str = "pred_Jss",
    xlabel: str = "Observed Jss",
    ylabel: str = "Predicted Jss",
    log_scale: bool = True,
    figsize: tuple = (6, 6),
    save: Optional[str] = None,
) -> None:
    """Scatter plot of observed vs predicted with identity line and metrics."""
    if plt is None:
        return

    obs = df[obs_col].dropna().values
    pred = df[pred_col].dropna().values
    n = min(len(obs), len(pred))
    obs, pred = obs[:n], pred[:n]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(obs, pred, s=40, alpha=0.7, edgecolors="black", linewidth=0.5)

    # Identity line
    lims = [min(obs.min(), pred.min()) * 0.8, max(obs.max(), pred.max()) * 1.2]
    ax.plot(lims, lims, "k--", alpha=0.5, label="Identity")
    # 2-fold lines
    ax.plot(lims, [l * 2 for l in lims], "r:", alpha=0.3, label="2-fold")
    ax.plot(lims, [l / 2 for l in lims], "r:", alpha=0.3)

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    # Metrics annotation
    r2 = 1 - np.sum((obs - pred) ** 2) / (np.sum((obs - obs.mean()) ** 2) + 1e-30)
    rmse = np.sqrt(np.mean((obs - pred) ** 2))
    ax.text(0.05, 0.92, f"R² = {r2:.3f}\nRMSE = {rmse:.3f}\nn = {n}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title("Observed vs. Predicted", fontsize=14)
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    plt.show()


def plot_calibration(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    n_bins: int = 10,
    figsize: tuple = (6, 6),
    save: Optional[str] = None,
) -> None:
    """Calibration plot: expected vs observed coverage at different confidence levels."""
    if plt is None:
        return

    quantiles = np.linspace(0.05, 0.95, n_bins)
    observed_fractions = []
    for q in quantiles:
        threshold = stats.norm.ppf(q) * y_pred_std + y_pred_mean
        frac = np.mean(y_true <= threshold)
        observed_fractions.append(frac)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(quantiles, observed_fractions, "o-", linewidth=2, label="Model")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.fill_between([0, 1], [0 - 0.05, 1 - 0.05], [0 + 0.05, 1 + 0.05],
                    alpha=0.1, color="gray")

    ax.set_xlabel("Expected quantile", fontsize=12)
    ax.set_ylabel("Observed fraction", fontsize=12)
    ax.set_title("Calibration Plot", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    plt.show()


def plot_blind_test_results(
    results_df: pd.DataFrame,
    endpoint: str = "Jss",
    figsize: tuple = (10, 5),
    save: Optional[str] = None,
) -> None:
    """Bar chart comparing blind-test performance across holdout categories."""
    if plt is None:
        return

    obs_col = f"observed_{endpoint}"
    pred_col = f"pred_{endpoint}"

    if obs_col not in results_df or pred_col not in results_df:
        return

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Scatter
    obs = results_df[obs_col].values
    pred = results_df[pred_col].values
    ax = axes[0]
    ax.scatter(obs, pred, s=40, alpha=0.7, edgecolors="black")
    lims = [min(obs.min(), pred.min()) * 0.5, max(obs.max(), pred.max()) * 2]
    ax.plot(lims, lims, "k--")
    ax.set_xlabel(f"Observed {endpoint}")
    ax.set_ylabel(f"Predicted {endpoint}")
    ax.set_title(f"Blind Test: {endpoint}")
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Fold error distribution
    ax2 = axes[1]
    fe = np.maximum(pred / (obs + 1e-30), obs / (pred + 1e-30))
    ax2.hist(fe, bins=20, edgecolor="black", alpha=0.7)
    ax2.axvline(2, color="red", linestyle="--", label="2-fold")
    ax2.axvline(3, color="orange", linestyle="--", label="3-fold")
    ax2.set_xlabel("Fold error")
    ax2.set_ylabel("Count")
    ax2.set_title("Fold Error Distribution")
    ax2.legend()

    plt.tight_layout()
    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    plt.show()
