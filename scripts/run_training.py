#!/usr/bin/env python3
"""
Train the state modulation network for transport prediction.

Key insight: Kp_standard = Jss / C_vehicle.
In PDE: C_donor_SC_surface = K_sv × C_vehicle.
Analytical: Kp = K_sv / R_total × unit_conversion.
"""

import sys, os, logging, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("training")

C_VEHICLE = 1000.0  # reference vehicle concentration, µg/cm³


def analytical_kp(D_sc, D_ve, K_sc_ve, logP, L_sc=15.0, L_ve=100.0, L_de=300.0):
    """Kp_standard (cm/h) = K_sv / R_total × unit_conversion.

    The series resistance gives flux from SC surface:
      J_skin = C_SC_surface / R_total
    And C_SC_surface = K_sv × C_vehicle.
    So Kp = Jss / C_vehicle = K_sv / R_total × 1e-4 × 3600.
    """
    D_dermis = max(D_ve * 2.0, 50.0)
    K_sv = np.clip(10 ** (0.69 * logP), 0.1, 500.0)

    R_sc = L_sc / (D_sc + 1e-10)
    R_ve = L_ve / (D_ve + 1e-10)
    R_de = L_de / (D_dermis + 1e-10)
    R_total = R_sc + K_sc_ve * R_ve + K_sc_ve * R_de  # s/µm

    kp = K_sv / (R_total + 1e-10) * 1e-4 * 3600.0  # cm/h
    return kp


def analytical_lag(D_sc, L_sc=15.0):
    """Lag time (h) dominated by SC."""
    return L_sc**2 / (6 * D_sc + 1e-10) / 3600.0


def find_optimal_params(mw, logp, target_kp, target_lag=None):
    """Find D_sc, D_ve, K_sc_ve that reproduce literature Kp."""
    from scipy.optimize import minimize

    K_sv = np.clip(10 ** (0.69 * logp), 0.1, 500.0)
    log_kp_pg = -2.7 + 0.71 * logp - 0.0061 * mw
    kp_pg = 10 ** log_kp_pg
    D_sc_init = max(kp_pg * 15e-4 / max(K_sv, 0.01) / 3600 * 1e8, 1e-6)
    D_ve_init = max(600 / (mw**0.5), D_sc_init * 20)
    K_sc_ve_init = np.clip(10**(0.39 * logp), 0.3, 50.0)

    x0 = np.array([np.log10(D_sc_init), np.log10(D_ve_init), np.log10(K_sc_ve_init)])

    def obj(x):
        kp = analytical_kp(10**x[0], 10**x[1], 10**x[2], logp)
        err = (np.log10(max(kp, 1e-15)) - np.log10(max(target_kp, 1e-15)))**2
        if target_lag and target_lag > 0:
            lag = analytical_lag(10**x[0])
            err += 0.05 * (np.log10(max(lag, 0.01)) - np.log10(max(target_lag, 0.01)))**2
        return err

    res = minimize(obj, x0, method="Nelder-Mead",
                   options={"maxiter": 300, "xatol": 1e-5, "fatol": 1e-8})
    kp_opt = analytical_kp(10**res.x[0], 10**res.x[1], 10**res.x[2], logp)
    lag_opt = analytical_lag(10**res.x[0])
    return res.x, kp_opt, lag_opt


def pde_kp_for_params(param_vec, logP, geom):
    """Full PDE → Kp (cm/h) = Jss / C_vehicle."""
    from virtual_skin.transport.layered_diffusion import (
        LayeredDiffusionPDE, TransportParameters,
    )
    D_sc, D_ve, K_sc_ve = 10**param_vec[0], 10**param_vec[1], 10**param_vec[2]
    K_sv = np.clip(10 ** (0.69 * logP), 0.1, 500.0)
    C_donor = C_VEHICLE * K_sv  # SC surface concentration

    tp = TransportParameters(
        D_sc=np.clip(D_sc, 1e-6, 1.0),
        D_ve=np.clip(D_ve, 0.01, 1000.0),
        D_dermis=max(D_ve * 2.0, 50.0),
        K_sc_ve=np.clip(K_sc_ve, 0.1, 100.0),
        k_bind_dermis=1e-5, k_clear_vasc=3e-4, w_appendage=1e-5,
        C_donor=C_donor,
    )
    try:
        pde = LayeredDiffusionPDE(geometry=geom, params=tp)
        result = pde.solve(t_total_s=24 * 3600, dt_output_s=1800)
        jss = pde.steady_state_flux(result)
        lag = pde.lag_time(result)
        kp = jss / C_VEHICLE  # Kp = Jss / C_vehicle
        return kp, lag
    except Exception:
        return 0.0, 0.0


def main():
    import torch
    from tqdm import tqdm
    os.makedirs("results/training", exist_ok=True)

    logger.info("Building expanded IVPT literature database...")
    from virtual_skin.data.public_datasets.huskindb_scraper import build_expanded_literature_db
    db = build_expanded_literature_db()

    from virtual_skin.data.drug_properties import DrugDescriptor
    from virtual_skin.transport.layered_diffusion import SkinLayerGeometry
    from virtual_skin.transport.state_modulation import StateModulationNetwork
    from virtual_skin.atlas.state_space import TissueStateVector

    geom = SkinLayerGeometry()
    healthy = TissueStateVector(0.8, 0.15, 0.2, 0.5, 0.3)

    # ═══════════════════════════════════════════════════════════
    # STAGE 1: Quick analytical test on caffeine to verify units
    # ═══════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("UNIT SANITY CHECK (caffeine)")
    x_test = np.array([np.log10(0.005), np.log10(30.0), np.log10(3.0)])
    kp_a = analytical_kp(0.005, 30.0, 3.0, logP=-0.07)
    kp_p, lag_p = pde_kp_for_params(x_test, -0.07, geom)
    logger.info(f"  Analytical Kp = {kp_a:.2e} cm/h")
    logger.info(f"  PDE Kp        = {kp_p:.2e} cm/h")
    logger.info(f"  PDE/Analytical ratio = {kp_p / max(kp_a, 1e-15):.2f}")
    logger.info(f"  Literature Kp = 1.72e-03 cm/h")

    # ═══════════════════════════════════════════════════════════
    # STAGE 2: Analytical optimization for all drugs
    # ═══════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2: Analytical optimization for all drugs")
    logger.info("=" * 60)

    calibration = []
    for drug_name in db.drug_names:
        kp = db.get_consensus_kp(drug_name)
        if kp is None or kp <= 0:
            continue
        records = db.get_drug_records(drug_name)
        r = records[0]
        lag = db.get_consensus_lag_time(drug_name)

        drug = DrugDescriptor(
            name=drug_name, molecular_weight=r.mw, logP=r.logp,
            solubility_mg_mL=r.donor_conc_mg_ml or 10.0,
        )

        x_opt, kp_opt, lag_opt = find_optimal_params(r.mw, r.logp, kp, lag)
        fold = max(kp_opt / max(kp, 1e-15), kp / max(kp_opt, 1e-15))

        calibration.append({
            "drug": drug, "drug_name": drug_name,
            "drug_vec": drug.to_vector(), "logp": r.logp,
            "target_kp": kp, "target_lag": lag,
            "optimal_params": x_opt,
            "kp_opt": kp_opt, "fold_opt": fold,
        })

    n = len(calibration)
    fold_s1 = [c["fold_opt"] for c in calibration]
    w3 = sum(1 for f in fold_s1 if f <= 3.0)
    logger.info(f"Optimized {n} drugs: {w3}/{n} within 3-fold")
    logger.info(f"Mean: {np.mean(fold_s1):.2f}x | Median: {np.median(fold_s1):.2f}x")

    # ═══════════════════════════════════════════════════════════
    # STAGE 3: PDE validation of analytical params (key subset)
    # ═══════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 3: PDE cross-validation (20 key drugs)")
    logger.info("=" * 60)

    key_drugs = ["caffeine", "hydrocortisone", "testosterone", "lidocaine",
                 "diclofenac", "nicotine", "estradiol", "fentanyl",
                 "water", "phenol", "naproxen", "progesterone",
                 "codeine", "diazepam", "minoxidil", "1_octanol",
                 "benzene", "sucrose", "methyl_paraben", "nitroglycerin"]

    pde_results = []
    for c in calibration:
        if c["drug_name"] not in key_drugs:
            continue
        drug = c["drug"]
        kp_pde, lag_pde = pde_kp_for_params(c["optimal_params"], drug.logP, geom)
        fold_pde = max(kp_pde / max(c["target_kp"], 1e-15),
                       c["target_kp"] / max(kp_pde, 1e-15))
        pde_results.append({
            "drug": drug.name, "kp_pde": kp_pde,
            "kp_analytical": c["kp_opt"], "kp_lit": c["target_kp"],
            "fold_pde": fold_pde,
        })
        logger.info(
            f"  {drug.name:25s} | lit={c['target_kp']:.2e} "
            f"ana={c['kp_opt']:.2e} pde={kp_pde:.2e} ({fold_pde:.1f}x)"
        )

    pde_folds = [r["fold_pde"] for r in pde_results]
    w3_pde = sum(1 for f in pde_folds if f <= 3.0)
    w5_pde = sum(1 for f in pde_folds if f <= 5.0)
    logger.info(f"\nPDE: {w3_pde}/{len(pde_results)} within 3-fold, "
                f"{w5_pde}/{len(pde_results)} within 5-fold")
    logger.info(f"Mean: {np.mean(pde_folds):.2f}x | Median: {np.median(pde_folds):.2f}x")

    # ═══════════════════════════════════════════════════════════
    # STAGE 4: Train network + final validation
    # ═══════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info(f"STAGE 4: Training network on {n} drugs")
    logger.info("=" * 60)

    n_drug_desc = len(calibration[0]["drug_vec"])
    net = StateModulationNetwork(n_state_axes=5, n_drug_desc=n_drug_desc,
                                 hidden_dims=[64, 32])

    state_vecs = torch.tensor(np.tile(healthy.to_array(), (n, 1)), dtype=torch.float32)
    drug_vecs = torch.tensor(np.stack([c["drug_vec"] for c in calibration]), dtype=torch.float32)
    target_params = torch.tensor(np.stack([c["optimal_params"] for c in calibration]), dtype=torch.float32)

    optimizer = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000)

    history = {"loss": []}
    best_loss, best_state = float("inf"), None

    for epoch in tqdm(range(10000), desc="ModNet training"):
        optimizer.zero_grad()
        params_dict = net(state_vecs, drug_vecs)
        pred_log = torch.stack([
            torch.log10(params_dict["D_sc"] + 1e-10),
            torch.log10(params_dict["D_ve"] + 1e-10),
            torch.log10(params_dict["K_sc_ve"] + 1e-10),
        ], dim=1)
        loss = torch.nn.functional.smooth_l1_loss(pred_log, target_params)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        history["loss"].append(loss.item())
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        if epoch % 1000 == 0:
            tqdm.write(f"  epoch {epoch:5d} | loss={loss.item():.6f}")

    if best_state:
        net.load_state_dict(best_state)

    # Final analytical validation
    net.eval()
    all_results = []
    for c in calibration:
        drug = c["drug"]
        params = net.predict_params(healthy, c["drug_vec"])
        D_sc = np.clip(params["D_sc"], 1e-6, 1.0)
        D_ve = np.clip(params["D_ve"], 0.01, 1000.0)
        K_sc_ve = np.clip(params["K_sc_ve"], 0.1, 100.0)
        kp_pred = analytical_kp(D_sc, D_ve, K_sc_ve, drug.logP)
        fold = max(kp_pred / max(c["target_kp"], 1e-15),
                    c["target_kp"] / max(kp_pred, 1e-15))
        all_results.append({
            "drug": drug.name, "mw": drug.molecular_weight, "logp": drug.logP,
            "kp_pred": kp_pred, "kp_lit": c["target_kp"], "fold_error": fold,
        })

    fold_errors = [r["fold_error"] for r in all_results]
    within_3 = sum(1 for f in fold_errors if f <= 3.0)
    within_5 = sum(1 for f in fold_errors if f <= 5.0)
    within_10 = sum(1 for f in fold_errors if f <= 10.0)

    logger.info(f"\n{'='*70}")
    logger.info("FINAL RESULTS (Network Analytical)")
    logger.info(f"{'='*70}")
    logger.info(f"Drugs tested:    {len(all_results)}")
    logger.info(f"Mean fold error: {np.mean(fold_errors):.2f}x")
    logger.info(f"Median fold:     {np.median(fold_errors):.2f}x")
    logger.info(f"Within 3-fold:   {within_3}/{len(all_results)} ({100*within_3/len(all_results):.0f}%)")
    logger.info(f"Within 5-fold:   {within_5}/{len(all_results)} ({100*within_5/len(all_results):.0f}%)")
    logger.info(f"Within 10-fold:  {within_10}/{len(all_results)} ({100*within_10/len(all_results):.0f}%)")

    # PDE final check on key drugs using network params
    logger.info(f"\nPDE Final (network predicted params):")
    pde_final = []
    for c in calibration:
        if c["drug_name"] not in key_drugs:
            continue
        drug = c["drug"]
        params = net.predict_params(healthy, c["drug_vec"])
        D_sc = np.clip(params["D_sc"], 1e-6, 1.0)
        D_ve = np.clip(params["D_ve"], 0.01, 1000.0)
        K_sc_ve = np.clip(params["K_sc_ve"], 0.1, 100.0)
        x = np.array([np.log10(D_sc), np.log10(D_ve), np.log10(K_sc_ve)])
        kp_pde, lag = pde_kp_for_params(x, drug.logP, geom)
        fold = max(kp_pde / max(c["target_kp"], 1e-15),
                    c["target_kp"] / max(kp_pde, 1e-15))
        pde_final.append({"drug": drug.name, "kp_pde": kp_pde, "kp_lit": c["target_kp"], "fold": fold})
        logger.info(f"  {drug.name:25s} | pred={kp_pde:.2e} lit={c['target_kp']:.2e} | fold={fold:.1f}x")

    pf = [r["fold"] for r in pde_final]
    w3f = sum(1 for f in pf if f <= 3.0)
    logger.info(f"\nPDE Final: {w3f}/{len(pde_final)} within 3-fold "
                f"(mean={np.mean(pf):.2f}x, median={np.median(pf):.2f}x)")

    # Save
    torch.save(net.state_dict(), "results/training/modulation_net.pt")
    logger.info("Saved model")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        ax = axes[0]
        ax.semilogy(history["loss"], alpha=0.2, color="blue")
        w = max(50, len(history["loss"]) // 50)
        sm = np.convolve(history["loss"], np.ones(w)/w, mode="valid")
        ax.semilogy(sm, color="blue", lw=2)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Training"); ax.grid(True, alpha=0.3)

        ax = axes[1]
        kpp = [r["kp_pred"] for r in all_results]
        kpl = [r["kp_lit"] for r in all_results]
        cols = ["#4CAF50" if r["fold_error"]<=3 else "#FF9800" if r["fold_error"]<=10 else "#F44336" for r in all_results]
        ax.scatter(kpl, kpp, s=30, c=cols, edgecolors="k", alpha=0.7, linewidths=0.5)
        vals = [v for v in kpp+kpl if v > 0]
        lo, hi = min(vals)*0.3, max(vals)*3
        ax.plot([lo,hi],[lo,hi],"k--",alpha=0.5,lw=2)
        ax.fill_between([lo,hi],[lo/3,hi/3],[lo*3,hi*3],alpha=0.1,color="green")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Literature Kp"); ax.set_ylabel("Predicted Kp")
        ax.set_title(f"{within_3}/{len(all_results)} within 3-fold")
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        logf = np.log10(fold_errors)
        ax.hist(logf, bins=25, color="#2196F3", edgecolor="k", alpha=0.7)
        ax.axvline(np.log10(3), color="green", ls="--", lw=2, label="3×")
        ax.set_xlabel("log₁₀(Fold Error)"); ax.set_ylabel("Count")
        ax.set_title(f"Median={np.median(fold_errors):.1f}×"); ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle(f"Virtual Skin: {len(all_results)} drugs | "
                     f"Within 3-fold: {within_3}/{len(all_results)} ({100*within_3/len(all_results):.0f}%)", fontsize=13)
        plt.tight_layout()
        plt.savefig("results/training/training_results.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved training_results.png")
    except Exception as e:
        logger.warning(f"Plot failed: {e}")


if __name__ == "__main__":
    main()
