"""
Training pipeline for the state-modulation network and PINN solver.

Two-stage approach:
  Stage 1: Train StateModulationNetwork on IVPT literature calibration data
            → maps (tissue_state, drug_features) → transport parameters
            → loss = |Jss_pred - Jss_lit|² + |lag_pred - lag_lit|² + prior
  Stage 2: Fine-tune with PINN physics loss
            → ensures PDE consistency at all (x, t) points

The key insight: we don't train the network on raw parameters (unobservable),
but on the *downstream pharmacokinetic endpoints* (Jss, lag, retention) which
are observable in IVPT experiments. The PDE solver acts as a differentiable
physics layer between parameters and endpoints.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from ..atlas.state_space import TissueStateVector
from ..data.drug_properties import DrugDescriptor
from .state_modulation import StateModulationNetwork, ParameterPrior
from .layered_diffusion import LayeredDiffusionPDE, SkinLayerGeometry, TransportParameters
from .pinn_solver import SkinPINNSolver

logger = logging.getLogger(__name__)


class CalibrationDataset:
    """Dataset of (drug, tissue_state, observed_endpoints) for training."""

    def __init__(self):
        self.entries: List[Dict[str, Any]] = []

    def add_entry(
        self,
        drug: DrugDescriptor,
        tissue_state: TissueStateVector,
        jss: float,
        lag_time: float = 0.0,
        retention: float = 0.0,
        weight: float = 1.0,
    ):
        self.entries.append({
            "drug": drug,
            "drug_vec": drug.to_vector(),
            "state_vec": tissue_state.to_array(),
            "jss": jss,
            "lag_time": lag_time,
            "retention": retention,
            "weight": weight,
        })

    @classmethod
    def from_literature_db(cls, db, drug_lib=None) -> "CalibrationDataset":
        """Build calibration dataset from IVPTLiteratureDB + DrugLibrary."""
        from ..data.drug_properties import DrugDescriptor, DrugLibrary
        from ..data.public_datasets.ivpt_literature import IVPTLiteratureDB

        if drug_lib is None:
            drug_lib = DrugLibrary.default_library()

        dataset = cls()
        healthy_state = TissueStateVector(
            barrier_integrity=0.8, inflammatory_load=0.15,
            ecm_remodeling=0.2, vascularization=0.5, appendage_openness=0.3,
        )

        for drug_name in db.drug_names:
            records = db.get_drug_records(drug_name)
            if not records:
                continue

            try:
                drug = drug_lib.get(drug_name)
            except KeyError:
                r = records[0]
                drug = DrugDescriptor(
                    name=drug_name,
                    molecular_weight=r.mw,
                    logP=r.logp,
                    solubility_mg_mL=r.donor_conc_mg_ml,
                )
                drug_lib.add(drug)

            consensus_jss = db.get_consensus_jss(drug_name)
            consensus_lag = db.get_consensus_lag_time(drug_name)

            if consensus_jss is not None and consensus_jss > 0:
                dataset.add_entry(
                    drug=drug,
                    tissue_state=healthy_state,
                    jss=consensus_jss,
                    lag_time=consensus_lag or 2.0,
                    weight=1.0,
                )

        logger.info(f"Built calibration dataset: {len(dataset.entries)} entries")
        return dataset

    def to_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert to batched tensors for training."""
        state_vecs = torch.tensor(
            np.stack([e["state_vec"] for e in self.entries]),
            dtype=torch.float32,
        )
        drug_vecs = torch.tensor(
            np.stack([e["drug_vec"] for e in self.entries]),
            dtype=torch.float32,
        )
        jss_targets = torch.tensor(
            [e["jss"] for e in self.entries], dtype=torch.float32,
        )
        lag_targets = torch.tensor(
            [e["lag_time"] for e in self.entries], dtype=torch.float32,
        )
        weights = torch.tensor(
            [e["weight"] for e in self.entries], dtype=torch.float32,
        )
        return state_vecs, drug_vecs, jss_targets, lag_targets, weights


class DifferentiablePDESolver:
    """Lightweight differentiable proxy for the PDE solver.

    Instead of running the full scipy.integrate solver (which isn't differentiable),
    we use an analytical approximation of Jss and lag time for the training loop,
    then fine-tune with the PINN.

    For a 3-layer skin with rate-limiting SC:
      Jss ≈ D_sc * C_donor / L_sc  (simplified)
      Jss ≈ C_donor / (L_sc/D_sc + L_ve*K_sc_ve/D_ve + L_dermis*K_sc_ve/D_dermis)
      lag ≈ L_sc²/(6*D_sc) + L_ve²/(6*D_ve) + ...
    """

    def __init__(self, geometry: Optional[SkinLayerGeometry] = None):
        self.geom = geometry or SkinLayerGeometry()

    def compute_jss_lag(
        self,
        params: Dict[str, torch.Tensor],
        C_donor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiable Jss and lag time from transport parameters.

        All diffusivities in µm²/s (matching PDE solver).
        C_donor in µg/cm³ (already includes SC/vehicle partition).
        Output: Jss in µg/cm²/h, lag in hours.
        """
        D_sc = params["D_sc"]             # µm²/s
        D_ve = params["D_ve"]             # µm²/s
        K_sc_ve = params["K_sc_ve"]       # dimensionless
        k_bind = params.get("k_bind_dermis", torch.zeros_like(D_sc))

        L_sc = self.geom.sc_thickness     # µm
        L_ve = self.geom.ve_thickness     # µm
        L_de = self.geom.dermis_thickness # µm

        D_dermis = D_ve * 2.0 + 1.0      # µm²/s

        # Series resistance model for multi-layer skin
        # R_i has units of [µm] / [µm²/s] = s/µm
        # At SC-VE interface, effective resistance includes partition:
        # flux continuity means J = D_sc*dC/dx = D_ve*dC/dx
        # The VE sees concentration C_ve = C_sc/K_sc_ve at interface
        R_sc = L_sc / (D_sc + 1e-10)
        R_ve = L_ve / (D_ve + 1e-10)
        R_de = L_de / (D_dermis + 1e-10)

        # For the series model through SC→VE→dermis:
        # C_sc_surface = C_donor (already partitioned into SC)
        # Effective total resistance accounts for partition drops
        # At SC-VE boundary: C_ve_top = C_sc_bottom / K_sc_ve
        # So total flux: J = C_donor / (R_sc + K_sc_ve * R_ve + K_sc_ve * R_de)
        R_total = R_sc + K_sc_ve * R_ve + K_sc_ve * R_de

        # J in units: [µg/cm³] / [s/µm] = µg·µm/(cm³·s)
        # Convert: 1 µm = 1e-4 cm → J * 1e-4 = µg/(cm²·s)
        # Then * 3600 for µg/(cm²·h)
        jss = C_donor / (R_total + 1e-10) * 1e-4 * 3600.0

        # Binding sink reduces effective Jss
        tau_bind = k_bind * L_de * L_de / (D_dermis + 1e-10)
        binding_factor = 1.0 / (1.0 + tau_bind)
        jss = jss * binding_factor

        # Lag time: L²/(6D) for rate-limiting layer (SC)
        lag_sc = L_sc ** 2 / (6.0 * D_sc + 1e-10)  # seconds
        lag_ve = L_ve ** 2 / (6.0 * D_ve + 1e-10)
        lag_de = L_de ** 2 / (6.0 * D_dermis + 1e-10)
        lag_total = lag_sc + lag_ve * 0.2 + lag_de * 0.05
        lag_h = lag_total / 3600.0

        return jss, lag_h


class StateModulationTrainer:
    """Train the StateModulationNetwork on IVPT calibration data.

    Uses a differentiable PDE proxy for gradient-based optimization,
    with the actual PDE solver for validation.
    """

    def __init__(
        self,
        net: Optional[StateModulationNetwork] = None,
        geometry: Optional[SkinLayerGeometry] = None,
        lr: float = 1e-3,
        epochs: int = 2000,
        prior_weight: float = 0.01,
        log_scale_loss: bool = True,
    ):
        self.net = net or StateModulationNetwork()
        self.geom = geometry or SkinLayerGeometry()
        self.proxy = DifferentiablePDESolver(self.geom)
        self.prior = ParameterPrior()
        self.lr = lr
        self.epochs = epochs
        self.prior_weight = prior_weight
        self.log_scale_loss = log_scale_loss

    def train(
        self,
        dataset: CalibrationDataset,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train the modulation network to match observed Jss and lag time."""
        state_vecs, drug_vecs, jss_targets, lag_targets, weights = dataset.to_tensors()
        n = len(dataset.entries)

        # Compute C_donor for each drug (with SC/vehicle partition)
        C_donors = []
        for entry in dataset.entries:
            drug = entry["drug"]
            C_vehicle = drug.solubility_mg_mL * 1000 if drug.solubility_mg_mL else 1000.0
            K_sv = np.clip(10 ** (0.69 * drug.logP), 0.1, 500.0)
            C_donors.append(C_vehicle * K_sv)
        C_donor_t = torch.tensor(C_donors, dtype=torch.float32)

        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        history = {"loss": [], "jss_loss": [], "lag_loss": [], "prior_loss": []}
        best_loss = float("inf")
        best_state = None

        iterator = tqdm(range(self.epochs), desc="ModNet training") if verbose else range(self.epochs)

        for epoch in iterator:
            optimizer.zero_grad()

            # Forward pass through modulation network
            params = self.net(state_vecs, drug_vecs)

            # Compute predicted Jss and lag through differentiable PDE proxy
            jss_pred, lag_pred = self.proxy.compute_jss_lag(params, C_donor_t)

            # Loss on Jss (log-scale for orders-of-magnitude range)
            if self.log_scale_loss:
                jss_loss = (weights * (
                    torch.log10(jss_pred.clamp(min=1e-6)) - torch.log10(jss_targets.clamp(min=1e-6))
                ) ** 2).mean()
            else:
                jss_loss = (weights * (jss_pred - jss_targets) ** 2).mean()

            # Loss on lag time
            lag_valid = lag_targets > 0
            if lag_valid.any():
                lag_loss = ((lag_pred[lag_valid] - lag_targets[lag_valid]) ** 2).mean() * 0.1
            else:
                lag_loss = torch.tensor(0.0)

            # Prior regularization
            prior_loss = -self.prior.log_prob(params) / n * self.prior_weight

            loss = jss_loss + lag_loss + prior_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

            history["loss"].append(loss.item())
            history["jss_loss"].append(jss_loss.item())
            history["lag_loss"].append(lag_loss.item())
            history["prior_loss"].append(prior_loss.item())

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}

            if verbose and epoch % 200 == 0:
                with torch.no_grad():
                    fold_errors = torch.abs(
                        torch.log10(jss_pred.clamp(min=1e-6)) - torch.log10(jss_targets.clamp(min=1e-6))
                    )
                    max_fold = 10 ** fold_errors.max().item()
                    mean_fold = 10 ** fold_errors.mean().item()
                tqdm.write(
                    f"  epoch {epoch:4d} | loss={loss.item():.4f} | "
                    f"mean fold={mean_fold:.2f}x | max fold={max_fold:.1f}x"
                )

        # Restore best model
        if best_state is not None:
            self.net.load_state_dict(best_state)

        return history

    def validate_with_pde(
        self,
        dataset: CalibrationDataset,
    ) -> Dict[str, Any]:
        """Validate predictions using the full PDE solver (not proxy)."""
        self.net.eval()
        results = []

        for entry in dataset.entries:
            drug = entry["drug"]
            sv = TissueStateVector(
                barrier_integrity=entry["state_vec"][0],
                inflammatory_load=entry["state_vec"][1],
                ecm_remodeling=entry["state_vec"][2],
                vascularization=entry["state_vec"][3],
                appendage_openness=entry["state_vec"][4],
            )

            params_dict = self.net.predict_params(sv, drug.to_vector())

            C_vehicle = drug.solubility_mg_mL * 1000 if drug.solubility_mg_mL else 1000.0
            K_sv = np.clip(10 ** (0.69 * drug.logP), 0.1, 500.0)

            # Network outputs are in µm²/s (matching PDE solver)
            D_sc = np.clip(params_dict.get("D_sc", 0.005), 1e-6, 1.0)
            D_ve = np.clip(params_dict.get("D_ve", 30.0), 0.01, 1000.0)
            D_dermis = max(D_ve * 2.0, 50.0)
            K_sc_ve = np.clip(params_dict.get("K_sc_ve", 3.0), 0.1, 100.0)

            tp = TransportParameters(
                D_sc=D_sc,
                D_ve=D_ve,
                D_dermis=D_dermis,
                K_sc_ve=K_sc_ve,
                k_bind_dermis=np.clip(params_dict.get("k_bind_dermis", 1e-5), 0.0, 0.001),
                k_clear_vasc=np.clip(params_dict.get("k_clear_vasc", 3e-4), 0.0, 0.005),
                w_appendage=np.clip(params_dict.get("w_appendage", 1e-5), 0.0, 0.001),
                C_donor=C_vehicle * K_sv,
            )

            pde = LayeredDiffusionPDE(geometry=self.geom, params=tp)
            try:
                result = pde.solve(t_total_s=24 * 3600, dt_output_s=1800)
                jss_pred = pde.steady_state_flux(result)
                lag_pred = pde.lag_time(result)
            except Exception:
                jss_pred = 0.0
                lag_pred = 0.0

            jss_lit = entry["jss"]
            fold_error = max(
                jss_pred / max(jss_lit, 1e-12),
                jss_lit / max(jss_pred, 1e-12),
            )

            results.append({
                "drug": drug.name,
                "jss_pred": jss_pred,
                "jss_lit": jss_lit,
                "fold_error": fold_error,
                "lag_pred": lag_pred,
                "lag_lit": entry["lag_time"],
                "params": params_dict,
            })

        fold_errors = [r["fold_error"] for r in results]
        within_3fold = sum(1 for f in fold_errors if f <= 3.0)

        summary = {
            "results": results,
            "mean_fold": float(np.mean(fold_errors)),
            "median_fold": float(np.median(fold_errors)),
            "max_fold": float(np.max(fold_errors)),
            "within_3fold": within_3fold,
            "total": len(results),
        }

        logger.info(
            f"PDE Validation: mean fold={summary['mean_fold']:.2f}x | "
            f"within 3-fold: {within_3fold}/{len(results)}"
        )
        for r in results:
            logger.info(
                f"  {r['drug']:20s} | pred={r['jss_pred']:8.3f} | "
                f"lit={r['jss_lit']:8.3f} | fold={r['fold_error']:.1f}x"
            )

        return summary


class PINNFineTuner:
    """Stage 2: Fine-tune predictions using PINN physics-informed refinement."""

    def __init__(
        self,
        geometry: Optional[SkinLayerGeometry] = None,
        epochs: int = 5000,
        lr: float = 5e-4,
    ):
        self.geom = geometry or SkinLayerGeometry()
        self.epochs = epochs
        self.lr = lr

    def fine_tune_for_drug(
        self,
        params: Dict[str, float],
        C_donor: float,
        observed_jss: float,
        observed_lag: float = 0.0,
        t_max_h: float = 24.0,
    ) -> Dict[str, Any]:
        """Run PINN for a specific drug × state to refine transport parameters."""
        pinn = SkinPINNSolver(
            geometry=self.geom,
            epochs=self.epochs,
            lr=self.lr,
            loss_weights={"pde": 1.0, "bc": 10.0, "ic": 5.0, "data": 100.0},
        )

        # Create observed data from Jss (steady-state flux at various times)
        t_obs_s = np.linspace(observed_lag * 3600, t_max_h * 3600, 20)
        x_obs = np.full_like(t_obs_s, self.geom.total_thickness - 1.0)
        C_gradient = observed_jss * 1e4 / (params.get("D_ve", 0.1) * 5) * 1.0
        C_obs = np.full_like(t_obs_s, max(C_gradient, 0.001))

        observed_data = {
            "x": x_obs.astype(np.float32),
            "time_s": t_obs_s.astype(np.float32),
            "concentration": C_obs.astype(np.float32),
        }

        history = pinn.train(
            params=params,
            t_max_s=t_max_h * 3600,
            C_donor=C_donor,
            observed_data=observed_data,
        )

        t_eval = np.linspace(0, t_max_h * 3600, 49)
        cum_perm = pinn.predict_cumulative_permeation(t_eval, params)

        return {
            "history": history,
            "cumulative_permeation": cum_perm,
            "time_h": t_eval / 3600.0,
            "pinn_solver": pinn,
        }


def run_full_training_pipeline(
    expanded_db=None,
    epochs_modnet: int = 3000,
    epochs_pinn: int = 3000,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the complete two-stage training pipeline.

    Returns trained network, validation results, and training history.
    """
    from ..data.drug_properties import DrugLibrary

    # Stage 0: Build expanded calibration dataset
    if expanded_db is None:
        try:
            from ..data.public_datasets.huskindb_scraper import build_expanded_literature_db
            expanded_db = build_expanded_literature_db()
        except Exception:
            from ..data.public_datasets.ivpt_literature import IVPTLiteratureDB
            expanded_db = IVPTLiteratureDB()

    logger.info(f"Calibration DB: {expanded_db.n_records} records, {len(expanded_db.drug_names)} drugs")

    drug_lib = DrugLibrary.default_library()
    cal_dataset = CalibrationDataset.from_literature_db(expanded_db, drug_lib)
    logger.info(f"Calibration dataset: {len(cal_dataset.entries)} drug × state entries")

    # Stage 1: Train StateModulationNetwork
    logger.info("=" * 60)
    logger.info("STAGE 1: Training State Modulation Network")
    logger.info("=" * 60)

    trainer = StateModulationTrainer(epochs=epochs_modnet, lr=2e-3, prior_weight=0.005)
    history = trainer.train(cal_dataset, verbose=verbose)

    # Validate with full PDE solver
    logger.info("\nValidating with full PDE solver...")
    val_results = trainer.validate_with_pde(cal_dataset)

    # Stage 2: PINN refinement (for drugs with >3-fold error)
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2: PINN Physics-Informed Refinement")
    logger.info("=" * 60)

    pinn_tuner = PINNFineTuner(geometry=trainer.geom, epochs=epochs_pinn)
    pinn_results = {}

    for r in val_results["results"]:
        if r["fold_error"] > 3.0 and r["jss_lit"] > 0.01:
            drug_name = r["drug"]
            logger.info(f"  PINN fine-tuning for {drug_name} (fold error: {r['fold_error']:.1f}x)")
            try:
                drug = drug_lib.get(drug_name)
                C_vehicle = drug.solubility_mg_mL * 1000 if drug.solubility_mg_mL else 1000.0
                K_sv = np.clip(10 ** (0.69 * drug.logP), 0.1, 500.0)

                pinn_result = pinn_tuner.fine_tune_for_drug(
                    params=r["params"],
                    C_donor=C_vehicle * K_sv,
                    observed_jss=r["jss_lit"],
                    observed_lag=r["lag_lit"],
                )
                pinn_results[drug_name] = pinn_result
                logger.info(f"    PINN final loss: {pinn_result['history']['total'][-1]:.4f}")
            except Exception as e:
                logger.warning(f"    PINN failed for {drug_name}: {e}")

    return {
        "modulation_net": trainer.net,
        "training_history": history,
        "validation": val_results,
        "pinn_results": pinn_results,
        "calibration_dataset": cal_dataset,
    }
