"""Transport model and pharmacokinetic endpoint visualisation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    plt = None

from ..transport.drug_transport import TransportPrediction


def plot_permeation_curve(
    predictions: List[TransportPrediction],
    observed: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    figsize: tuple = (10, 6),
    title: str = "Cumulative Permeation",
    save: Optional[str] = None,
) -> None:
    """Plot cumulative permeation Q(t) for one or more predictions + optional observed data."""
    if plt is None:
        raise ImportError("matplotlib required")

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(predictions), 10)))

    for i, pred in enumerate(predictions):
        label = f"{pred.drug_name} (Jss={pred.steady_state_flux:.2f})"
        ax.plot(pred.time_h, pred.cumulative_permeation, color=colors[i], label=label, linewidth=2)

    if observed:
        for i, (t_obs, q_obs) in enumerate(observed):
            ax.scatter(t_obs, q_obs, color=colors[i], marker="o", s=40,
                       edgecolors="black", zorder=5, label=f"Observed {i+1}")

    ax.set_xlabel("Time (h)", fontsize=12)
    ax.set_ylabel("Cumulative Permeation (µg/cm²)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    plt.show()


def plot_concentration_profile(
    prediction: TransportPrediction,
    time_indices: Optional[List[int]] = None,
    figsize: tuple = (10, 6),
    save: Optional[str] = None,
) -> None:
    """Plot concentration C(x) at selected time points, with layer shading."""
    if plt is None or prediction.concentration_profile is None:
        return

    from ..transport.layered_diffusion import SkinLayerGeometry
    geom = SkinLayerGeometry()

    fig, ax = plt.subplots(figsize=figsize)

    # Layer background shading
    ax.axvspan(0, geom.sc_thickness, alpha=0.15, color="gold", label="SC")
    ax.axvspan(geom.sc_thickness, geom.sc_thickness + geom.ve_thickness,
               alpha=0.15, color="salmon", label="VE")
    ax.axvspan(geom.sc_thickness + geom.ve_thickness, geom.total_thickness,
               alpha=0.15, color="skyblue", label="Dermis")

    n_times = prediction.concentration_profile.shape[0]
    if time_indices is None:
        time_indices = np.linspace(0, n_times - 1, min(8, n_times), dtype=int).tolist()

    x = np.linspace(0, geom.total_thickness, prediction.concentration_profile.shape[1])
    cmap = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))

    for j, ti in enumerate(time_indices):
        t_h = prediction.time_h[ti]
        ax.plot(x, prediction.concentration_profile[ti], color=cmap[j],
                label=f"t = {t_h:.1f} h", linewidth=1.5)

    ax.set_xlabel("Depth (µm)", fontsize=12)
    ax.set_ylabel("Concentration", fontsize=12)
    ax.set_title(f"Concentration Profile — {prediction.drug_name}", fontsize=14)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    plt.show()


def plot_sensitivity(
    sweep_result: Dict[str, np.ndarray],
    figsize: tuple = (12, 4),
    save: Optional[str] = None,
) -> None:
    """Plot endpoint sensitivity to a state axis sweep."""
    if plt is None:
        return

    axis_name = [k for k in sweep_result if k not in ("Jss", "lag_time", "target_AUC")][0]
    x = sweep_result[axis_name]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, (ep, ylabel) in zip(axes, [
        ("Jss", "Steady-state flux"),
        ("lag_time", "Lag time (h)"),
        ("target_AUC", "Target-layer AUC"),
    ]):
        ax.plot(x, sweep_result[ep], "o-", linewidth=2, markersize=4)
        ax.set_xlabel(axis_name, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Sensitivity to {axis_name}", fontsize=13)
    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    plt.show()
