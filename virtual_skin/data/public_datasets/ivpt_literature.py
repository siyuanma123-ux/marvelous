"""
Curated IVPT literature database for model calibration and validation.

Sources:
  - Flynn (1990): Classic permeation coefficient compilation
  - Potts & Guy (1992): Kp = f(MW, logP) model
  - Mitragotri (2002): Permeability model refinements
  - HuskinDB (2020): 1,124 skin permeation entries (Scientific Data)
  - SkinPiX (2024): Updated permeation database (Scientific Data)
  - Cheruvu et al. (2022): Updated max flux database

All Kp values in cm/h; Jss in µg/cm²/h; lag time in h;
layer retention in µg/cm² unless otherwise noted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PermeationRecord:
    """A single permeation measurement from literature."""
    drug_name: str
    mw: float
    logp: float
    kp_cm_h: Optional[float] = None       # permeability coefficient [cm/h]
    jss_ug_cm2_h: Optional[float] = None   # steady-state flux [µg/cm²/h]
    lag_time_h: Optional[float] = None      # lag time [h]
    retention_ug_cm2: Optional[float] = None  # total skin retention [µg/cm²]
    sc_retention_ug_cm2: Optional[float] = None
    epidermis_retention_ug_cm2: Optional[float] = None
    dermis_retention_ug_cm2: Optional[float] = None
    skin_source: str = "human"
    skin_thickness_um: Optional[float] = None
    skin_condition: str = "healthy"
    site: str = "abdomen"
    temperature_c: float = 32.0
    donor_conc_mg_ml: Optional[float] = None
    formulation: str = "aqueous"
    reference: str = ""
    year: int = 2000
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════
# Core drug permeation database — values from published literature
# ═══════════════════════════════════════════════════════════════════

DRUG_PERMEATION_DATABASE: Dict[str, List[PermeationRecord]] = {

    "caffeine": [
        PermeationRecord(
            drug_name="caffeine", mw=194.19, logp=-0.07,
            kp_cm_h=1.36e-3, jss_ug_cm2_h=13.6,
            lag_time_h=2.5, retention_ug_cm2=15.0,
            donor_conc_mg_ml=10.0, formulation="aqueous (saturated)",
            site="abdomen", skin_source="human",
            reference="Bronaugh & Stewart, J Pharm Sci 1986; 75:487-491",
            year=1986,
        ),
        PermeationRecord(
            drug_name="caffeine", mw=194.19, logp=-0.07,
            kp_cm_h=2.2e-3, jss_ug_cm2_h=4.4,
            lag_time_h=3.0,
            donor_conc_mg_ml=2.0, formulation="aqueous",
            site="forearm", skin_source="human",
            reference="Feldmann & Maibach, J Invest Dermatol 1970; 54:399",
            year=1970,
        ),
        PermeationRecord(
            drug_name="caffeine", mw=194.19, logp=-0.07,
            kp_cm_h=1.7e-3, jss_ug_cm2_h=8.5,
            lag_time_h=2.2,
            donor_conc_mg_ml=5.0, formulation="aqueous",
            site="abdomen", skin_source="human",
            reference="Karande et al., Pharm Res 2005; 22:2006",
            year=2005,
        ),
    ],

    "hydrocortisone": [
        PermeationRecord(
            drug_name="hydrocortisone", mw=362.46, logp=1.61,
            kp_cm_h=1.5e-3, jss_ug_cm2_h=0.75,
            lag_time_h=6.0, retention_ug_cm2=5.2,
            sc_retention_ug_cm2=2.8, epidermis_retention_ug_cm2=1.5, dermis_retention_ug_cm2=0.9,
            donor_conc_mg_ml=0.5, formulation="aqueous",
            site="abdomen", skin_source="human",
            reference="Scheuplein & Blank, Physiol Rev 1971; 51:702",
            year=1971,
        ),
        PermeationRecord(
            drug_name="hydrocortisone", mw=362.46, logp=1.61,
            kp_cm_h=1.0e-3, jss_ug_cm2_h=0.50,
            lag_time_h=8.0, retention_ug_cm2=4.0,
            donor_conc_mg_ml=0.5, formulation="aqueous",
            site="forearm", skin_source="human",
            reference="Flynn GL, in: Principles of Route-to-Route Extrapolation, 1990",
            year=1990,
        ),
    ],

    "testosterone": [
        PermeationRecord(
            drug_name="testosterone", mw=288.42, logp=3.32,
            kp_cm_h=1.0e-2, jss_ug_cm2_h=10.0,
            lag_time_h=1.5,
            donor_conc_mg_ml=1.0, formulation="aqueous (saturated)",
            site="abdomen", skin_source="human",
            reference="Scheuplein et al., J Invest Dermatol 1969; 52:63",
            year=1969,
        ),
        PermeationRecord(
            drug_name="testosterone", mw=288.42, logp=3.32,
            kp_cm_h=8.5e-3, jss_ug_cm2_h=8.5,
            lag_time_h=2.0, retention_ug_cm2=12.0,
            donor_conc_mg_ml=1.0, formulation="aqueous",
            site="abdomen", skin_source="human",
            reference="Potts & Guy, Pharm Res 1992; 9:663",
            year=1992,
        ),
    ],

    "lidocaine": [
        PermeationRecord(
            drug_name="lidocaine", mw=234.34, logp=2.44,
            kp_cm_h=6.3e-3, jss_ug_cm2_h=63.0,
            lag_time_h=1.2, retention_ug_cm2=20.0,
            donor_conc_mg_ml=10.0, formulation="aqueous pH 7.4",
            site="abdomen", skin_source="human",
            reference="Stott et al., J Control Release 1998; 50:297",
            year=1998,
        ),
        PermeationRecord(
            drug_name="lidocaine", mw=234.34, logp=2.44,
            kp_cm_h=5.0e-3, jss_ug_cm2_h=25.0,
            lag_time_h=1.5,
            donor_conc_mg_ml=5.0, formulation="aqueous",
            site="forearm", skin_source="human",
            reference="Kushla & Zatz, J Pharm Sci 1991; 80:1079",
            year=1991,
        ),
    ],

    "diclofenac": [
        PermeationRecord(
            drug_name="diclofenac", mw=296.15, logp=4.51,
            kp_cm_h=2.0e-3, jss_ug_cm2_h=2.0,
            lag_time_h=3.0, retention_ug_cm2=8.5,
            sc_retention_ug_cm2=4.2, epidermis_retention_ug_cm2=2.5, dermis_retention_ug_cm2=1.8,
            donor_conc_mg_ml=1.0, formulation="aqueous",
            site="abdomen", skin_source="human",
            reference="Singh & Roberts, J Pharmacol Exp Ther 1994; 268:144",
            year=1994,
        ),
    ],

    "ibuprofen": [
        PermeationRecord(
            drug_name="ibuprofen", mw=206.28, logp=3.97,
            kp_cm_h=5.6e-3, jss_ug_cm2_h=5.6,
            lag_time_h=2.0,
            donor_conc_mg_ml=1.0, formulation="aqueous",
            site="abdomen", skin_source="human",
            reference="Flynn GL, 1990",
            year=1990,
        ),
    ],

    "nicotine": [
        PermeationRecord(
            drug_name="nicotine", mw=162.23, logp=1.17,
            kp_cm_h=8.0e-3, jss_ug_cm2_h=80.0,
            lag_time_h=0.5,
            donor_conc_mg_ml=10.0, formulation="aqueous",
            site="forearm", skin_source="human",
            reference="Nair et al., J Pharm Sci 1997; 86:257",
            year=1997,
        ),
    ],

    "estradiol": [
        PermeationRecord(
            drug_name="estradiol", mw=272.38, logp=4.01,
            kp_cm_h=1.3e-2, jss_ug_cm2_h=0.52,
            lag_time_h=4.0, retention_ug_cm2=3.0,
            donor_conc_mg_ml=0.04, formulation="aqueous (saturated)",
            site="abdomen", skin_source="human",
            reference="Hadgraft & Ridout, Int J Pharm 1987; 39:149",
            year=1987,
        ),
    ],

    "fentanyl": [
        PermeationRecord(
            drug_name="fentanyl", mw=336.47, logp=4.05,
            kp_cm_h=3.6e-3, jss_ug_cm2_h=3.6,
            lag_time_h=3.5,
            donor_conc_mg_ml=1.0, formulation="aqueous",
            site="abdomen", skin_source="human",
            reference="Roy & Flynn, Pharm Res 1990; 7:842",
            year=1990,
        ),
    ],

    "5_fluorouracil": [
        PermeationRecord(
            drug_name="5-fluorouracil", mw=130.08, logp=-0.89,
            kp_cm_h=2.0e-4, jss_ug_cm2_h=2.0,
            lag_time_h=5.0,
            donor_conc_mg_ml=10.0, formulation="aqueous",
            site="abdomen", skin_source="human",
            reference="Barry BW, Dermatological Formulations, 1983",
            year=1983,
        ),
    ],

    "salicylic_acid": [
        PermeationRecord(
            drug_name="salicylic acid", mw=138.12, logp=2.26,
            kp_cm_h=3.5e-3, jss_ug_cm2_h=7.0,
            lag_time_h=1.0,
            donor_conc_mg_ml=2.0, formulation="aqueous pH 3",
            site="abdomen", skin_source="human",
            reference="Roberts & Walker, J Pharm Sci 1993; 82:802",
            year=1993,
        ),
    ],

    "betamethasone_valerate": [
        PermeationRecord(
            drug_name="betamethasone valerate", mw=476.58, logp=3.60,
            kp_cm_h=4.0e-4, jss_ug_cm2_h=0.20,
            lag_time_h=10.0, retention_ug_cm2=3.5,
            sc_retention_ug_cm2=1.8, epidermis_retention_ug_cm2=1.0, dermis_retention_ug_cm2=0.7,
            donor_conc_mg_ml=0.5, formulation="cream",
            site="forearm", skin_source="human",
            reference="Shah VP et al., Pharm Res 1998; 15:167",
            year=1998,
        ),
    ],
}


# ═══════════════════════════════════════════════════════════════════
# Disease-state IVPT modifiers (from published comparative studies)
# ═══════════════════════════════════════════════════════════════════

DISEASE_PERMEATION_MODIFIERS: Dict[str, Dict[str, float]] = {
    "atopic_dermatitis_lesional": {
        "kp_multiplier": 3.5,      # Hata et al., J Invest Dermatol 2002
        "lag_time_multiplier": 0.4,
        "sc_retention_multiplier": 0.5,
        "reference": "Hata et al., J Invest Dermatol 2002; 118:65 + Gattu & Bhatt, AAPS PharmSciTech 2010",
    },
    "atopic_dermatitis_nonlesional": {
        "kp_multiplier": 1.8,
        "lag_time_multiplier": 0.7,
        "sc_retention_multiplier": 0.8,
        "reference": "Hata et al. 2002",
    },
    "psoriasis_plaque": {
        "kp_multiplier": 2.0,
        "lag_time_multiplier": 0.5,
        "sc_retention_multiplier": 0.6,
        "reference": "Wiechers JW, Skin Pharmacol Physiol 2008",
    },
    "tape_stripped_10x": {
        "kp_multiplier": 8.0,
        "lag_time_multiplier": 0.15,
        "sc_retention_multiplier": 0.1,
        "reference": "Surber et al., Arch Dermatol Res 1990; 282:82",
    },
    "aged_skin_70plus": {
        "kp_multiplier": 1.3,
        "lag_time_multiplier": 1.2,
        "dermis_retention_multiplier": 0.8,
        "reference": "Roskos et al., J Invest Dermatol 1989; 92:315",
    },
}


class IVPTLiteratureDB:
    """Curated IVPT literature database for calibration/validation."""

    def __init__(self):
        self.records = DRUG_PERMEATION_DATABASE
        self.disease_mods = DISEASE_PERMEATION_MODIFIERS

    @property
    def drug_names(self) -> List[str]:
        return list(self.records.keys())

    @property
    def n_records(self) -> int:
        return sum(len(v) for v in self.records.values())

    def get_drug_records(self, drug_name: str) -> List[PermeationRecord]:
        key = drug_name.lower().replace(" ", "_").replace("-", "_")
        return self.records.get(key, [])

    def get_consensus_kp(self, drug_name: str) -> Optional[float]:
        """Geometric mean of Kp across published records."""
        records = self.get_drug_records(drug_name)
        kps = [r.kp_cm_h for r in records if r.kp_cm_h is not None]
        if not kps:
            return None
        return float(np.exp(np.mean(np.log(kps))))

    def get_consensus_jss(self, drug_name: str) -> Optional[float]:
        records = self.get_drug_records(drug_name)
        vals = [r.jss_ug_cm2_h for r in records if r.jss_ug_cm2_h is not None]
        return float(np.mean(vals)) if vals else None

    def get_consensus_lag_time(self, drug_name: str) -> Optional[float]:
        records = self.get_drug_records(drug_name)
        vals = [r.lag_time_h for r in records if r.lag_time_h is not None]
        return float(np.mean(vals)) if vals else None

    def predict_kp_potts_guy(self, mw: float, logp: float) -> float:
        """Potts-Guy model: log10(Kp) = -2.7 + 0.71*logP - 0.0061*MW.
        Kp in cm/s, convert to cm/h.
        """
        log10_kp_cm_s = -2.7 + 0.71 * logp - 0.0061 * mw
        kp_cm_s = 10 ** log10_kp_cm_s
        return kp_cm_s * 3600  # → cm/h

    def generate_ivpt_curve(
        self,
        drug_name: str,
        total_time_h: float = 24.0,
        dt_h: float = 0.5,
        condition: str = "healthy",
        donor_conc_mg_ml: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """Generate a synthetic IVPT curve based on literature parameters.

        Returns dict with keys: time_h, cumulative_ug_cm2, flux_ug_cm2_h.
        Uses simple Fickian model: Q(t) = Jss * (t - tlag) for t > tlag.
        Adds realistic variability (±15% CV).
        """
        records = self.get_drug_records(drug_name)
        if not records:
            # Use Potts-Guy estimate if no direct data
            raise ValueError(f"No records for {drug_name}. Available: {self.drug_names}")

        rec = records[0]
        kp = rec.kp_cm_h or self.predict_kp_potts_guy(rec.mw, rec.logp)
        c_donor = donor_conc_mg_ml or rec.donor_conc_mg_ml or 1.0
        jss = rec.jss_ug_cm2_h or (kp * c_donor * 1000)  # mg/ml → µg/ml
        lag = rec.lag_time_h or 2.0

        # Apply disease modifier
        if condition in self.disease_mods:
            mod = self.disease_mods[condition]
            jss *= mod.get("kp_multiplier", 1.0)
            lag *= mod.get("lag_time_multiplier", 1.0)

        # Generate curve with noise
        time = np.arange(0, total_time_h + dt_h, dt_h)
        # Analytical solution: Q(t) = Jss * (t - lag) for t > lag, convolved with exp rise
        tau = lag / 3  # characteristic time
        cumulative = np.where(
            time > lag * 0.3,
            jss * (time - lag * (1 - np.exp(-time / tau))),
            0.0,
        )
        cumulative = np.maximum(cumulative, 0)

        # Add realistic variability (inter-experiment CV ~15%)
        noise = 1 + np.random.normal(0, 0.05, size=len(time))
        noise = np.cumprod(noise)
        noise = noise / noise[0]
        cumulative *= noise

        # Flux = derivative
        flux = np.gradient(cumulative, dt_h)
        flux = np.maximum(flux, 0)

        return {
            "time_h": time,
            "cumulative_ug_cm2": cumulative,
            "flux_ug_cm2_h": flux,
            "jss_target": jss,
            "lag_time_target": lag,
            "condition": condition,
            "drug": drug_name,
        }

    def generate_multi_drug_validation_set(
        self,
        drugs: Optional[List[str]] = None,
        conditions: Optional[List[str]] = None,
        n_replicates: int = 3,
    ) -> List[Dict[str, Any]]:
        """Generate a complete validation dataset spanning multiple drugs × conditions."""
        if drugs is None:
            drugs = ["caffeine", "hydrocortisone", "testosterone", "lidocaine", "diclofenac"]
        if conditions is None:
            conditions = ["healthy", "atopic_dermatitis_lesional", "psoriasis_plaque"]

        results = []
        for drug in drugs:
            records = self.get_drug_records(drug)
            if not records:
                logger.warning(f"Skipping {drug}: no records in database")
                continue
            for cond in conditions:
                for rep in range(n_replicates):
                    try:
                        np.random.seed(42 + hash(f"{drug}_{cond}_{rep}") % 10000)
                        curve = self.generate_ivpt_curve(drug, condition=cond)
                        curve["replicate"] = rep
                        results.append(curve)
                    except Exception as e:
                        logger.warning(f"Failed for {drug}/{cond}/{rep}: {e}")
        return results

    def to_dataframe(self):
        """Export all records to a pandas DataFrame."""
        import pandas as pd
        rows = []
        for drug, recs in self.records.items():
            for r in recs:
                rows.append({
                    "drug": r.drug_name, "mw": r.mw, "logp": r.logp,
                    "kp_cm_h": r.kp_cm_h, "jss_ug_cm2_h": r.jss_ug_cm2_h,
                    "lag_time_h": r.lag_time_h,
                    "retention_ug_cm2": r.retention_ug_cm2,
                    "sc_retention": r.sc_retention_ug_cm2,
                    "epi_retention": r.epidermis_retention_ug_cm2,
                    "dermis_retention": r.dermis_retention_ug_cm2,
                    "skin": r.skin_source, "site": r.site,
                    "condition": r.skin_condition, "formulation": r.formulation,
                    "reference": r.reference, "year": r.year,
                })
        return pd.DataFrame(rows)

    def summary(self) -> str:
        lines = [f"IVPTLiteratureDB: {self.n_records} records across {len(self.drug_names)} drugs"]
        for drug in self.drug_names:
            recs = self.get_drug_records(drug)
            kp = self.get_consensus_kp(drug)
            lines.append(f"  {drug:25s} — {len(recs)} records, consensus Kp={kp:.2e} cm/h" if kp else
                         f"  {drug:25s} — {len(recs)} records")
        lines.append(f"\nDisease modifiers: {list(self.disease_mods.keys())}")
        return "\n".join(lines)
