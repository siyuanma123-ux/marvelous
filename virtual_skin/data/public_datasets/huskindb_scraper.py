"""
Scraper and curated data from HuskinDB, SkinPiX, and other published
skin permeation databases.

Data sources:
  - HuskinDB (Stepanov et al., Sci Data 2020; doi:10.1038/s41597-020-00764-z)
    546 permeability values for 251 compounds; available at
    https://huskindb.drug-design.de and https://doi.org/10.7303/syn21998881
  - SkinPiX (Champmartin et al., Sci Data 2024; doi:10.1038/s41597-024-03026-4)
    Updated permeability data 2012-2021; curated subset at
    https://doi.org/10.57745/ZUU1DH (207 compounds with consensus Kp)
  - Cheruvu et al. (2022): Updated max flux / Kp dataset from Mendeley
    https://data.mendeley.com/datasets/8bs7hb2wj2/1
  - Flynn (1990): Classic compilation of ~97 compounds
  - Potts & Guy (1992): Validation / regression set

All Kp values are in cm/h from aqueous vehicle through human skin (in vitro).
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import URLError

from .ivpt_literature import IVPTLiteratureDB, PermeationRecord

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# Remote endpoints for the three public databases
# ═══════════════════════════════════════════════════════════════════════

HUSKINDB_URLS = {
    "website": "https://huskindb.drug-design.de",
    "synapse": "https://doi.org/10.7303/syn21998881",
    "data_page": "https://huskindb.drug-design.de/data/",
}

SKINPIX_URLS = {
    "paper": "https://doi.org/10.1038/s41597-024-03026-4",
    "data_gouv_huskin": (
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/"
        "doi:10.57745/ERDWZS"
    ),
    "data_gouv_skinpix": (
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/"
        "doi:10.57745/YATTFB"
    ),
}

CHERUVU_URLS = {
    "mendeley": "https://data.mendeley.com/datasets/8bs7hb2wj2/1",
}


# ═══════════════════════════════════════════════════════════════════════
# Manually curated permeation records from published compilations
#
# Primary data provenance
# ───────────────────────
#   [Flynn90]   Flynn GL (1990) Physicochemical determinants of skin
#               absorption. In: Gerrity TR, Henry CJ (eds) Principles
#               of Route-to-Route Extrapolation for Risk Assessment.
#               Elsevier, pp 93-127.
#   [PG92]      Potts RO & Guy RH (1992) Predicting skin permeability.
#               Pharm Res 9:663-669.
#   [M02]       Mitragotri S (2002) A theoretical analysis … J Pharm Sci
#               91:744-752.
#   [HDB20]     Stepanov et al., Sci Data 2020 (HuskinDB v1.01)
#   [SPX24]     Champmartin et al., Sci Data 2024 (SkinPiX)
#   [Cheruvu22] Cheruvu HS et al., Adv Drug Deliv Rev 2022.
#   [S&B71]     Scheuplein RJ & Blank IH, Physiol Rev 1971; 51:702-747.
#   [S69]       Scheuplein et al., J Invest Dermatol 1969; 52:63-70.
#   [B&S86]     Bronaugh RL & Stewart RF, J Pharm Sci 1986; 75:487-491.
#   [R&W93]     Roberts MS & Walker M, J Pharm Sci 1993; 82:802-808.
#   [K&Z91]     Kushla GP & Zatz JL, J Pharm Sci 1991; 80:1079-1083.
#   [Bur92]     Burch GE & Winsor T, Arch Intern Med 1946; 74:437 (water TEWL)
#   [John73]    Johnson ME et al., J Invest Dermatol 1973 (alcohols)
# ═══════════════════════════════════════════════════════════════════════

CURATED_PERMEATION_RECORDS: List[PermeationRecord] = [

    # ── Alcohols & small solvents (Flynn 1990 / Scheuplein 1971) ──────
    PermeationRecord(
        drug_name="water", mw=18.02, logp=-1.38,
        kp_cm_h=1.0e-3,
        reference="Flynn 1990; Scheuplein & Blank, Physiol Rev 1971; 51:702",
        year=1990, formulation="neat / aqueous",
        notes="Classic reference value for water permeability through human SC",
    ),
    PermeationRecord(
        drug_name="methanol", mw=32.04, logp=-0.77,
        kp_cm_h=1.4e-3,
        reference="Flynn 1990; Scheuplein & Blank 1971",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="ethanol", mw=46.07, logp=-0.31,
        kp_cm_h=8.0e-4,
        reference="Flynn 1990; Scheuplein & Blank 1971",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="2-propanol", mw=60.10, logp=0.05,
        kp_cm_h=1.8e-3,
        reference="Flynn 1990; Scheuplein & Blank 1971",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="1-butanol", mw=74.12, logp=0.88,
        kp_cm_h=6.3e-3,
        reference="Flynn 1990; Scheuplein & Blank 1971",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="1-pentanol", mw=88.15, logp=1.56,
        kp_cm_h=2.5e-2,
        reference="Flynn 1990; Scheuplein et al. 1969",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="1-hexanol", mw=102.17, logp=2.03,
        kp_cm_h=4.2e-2,
        reference="Flynn 1990; Scheuplein et al. 1969",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="1-heptanol", mw=116.20, logp=2.62,
        kp_cm_h=5.8e-2,
        reference="Flynn 1990; Scheuplein et al. 1969",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="1-octanol", mw=130.23, logp=3.15,
        kp_cm_h=1.6e-1,
        reference="Flynn 1990; Scheuplein et al. 1969",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="1-nonanol", mw=144.25, logp=3.77,
        kp_cm_h=1.4e-1,
        reference="Flynn 1990; Scheuplein et al. 1969",
        year=1990, formulation="aqueous",
        notes="Kp plateaus for higher n-alkanols",
    ),
    PermeationRecord(
        drug_name="1-decanol", mw=158.28, logp=4.57,
        kp_cm_h=5.3e-2,
        reference="Flynn 1990; Scheuplein et al. 1969",
        year=1990, formulation="aqueous",
        notes="Kp decreasing — aqueous solubility limitation",
    ),

    # ── Phenols (Flynn 1990 / HuskinDB) ──────────────────────────────
    PermeationRecord(
        drug_name="phenol", mw=94.11, logp=1.46,
        kp_cm_h=8.2e-3,
        reference="Flynn 1990; Roberts & Anderson, Pharm Res 1990",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="p-cresol", mw=108.14, logp=1.94,
        kp_cm_h=1.5e-2,
        reference="Flynn 1990; Roberts & Anderson 1990",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="resorcinol", mw=110.11, logp=0.80,
        kp_cm_h=8.7e-4,
        reference="Flynn 1990; Roberts & Anderson 1990",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="2,4-dichlorophenol", mw=163.00, logp=3.06,
        kp_cm_h=1.5e-2,
        reference="Flynn 1990; Roberts et al.",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="thymol", mw=150.22, logp=3.30,
        kp_cm_h=1.1e-1,
        reference="Flynn 1990; HuskinDB",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="chlorocresol", mw=142.58, logp=3.10,
        kp_cm_h=6.7e-2,
        reference="Flynn 1990; Roberts et al.",
        year=1990, formulation="aqueous",
    ),

    # ── Acids & related (Flynn 1990 / HuskinDB / Potts & Guy) ────────
    PermeationRecord(
        drug_name="benzoic acid", mw=122.12, logp=1.87,
        kp_cm_h=3.6e-3,
        reference="Flynn 1990; Potts & Guy 1992; Roberts & Walker 1993",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="nicotinic acid", mw=123.11, logp=0.36,
        kp_cm_h=5.3e-4,
        reference="Flynn 1990",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="nicotinamide", mw=122.12, logp=-0.37,
        kp_cm_h=5.2e-4,
        reference="Flynn 1990; Potts & Guy 1992",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="succinic acid", mw=118.09, logp=-0.59,
        kp_cm_h=3.0e-5,
        reference="Flynn 1990",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="hippuric acid", mw=179.17, logp=0.31,
        kp_cm_h=4.2e-4,
        reference="Flynn 1990",
        year=1990, formulation="aqueous",
    ),

    # ── NSAIDs (Flynn 1990, HuskinDB, SkinPiX, various) ─────────────
    PermeationRecord(
        drug_name="naproxen", mw=230.26, logp=3.18,
        kp_cm_h=4.8e-3, jss_ug_cm2_h=2.4,
        lag_time_h=2.5,
        reference="Flynn 1990; Potts & Guy 1992; Cheruvu et al. 2022",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="piroxicam", mw=331.35, logp=3.06,
        kp_cm_h=3.4e-4, jss_ug_cm2_h=0.34,
        lag_time_h=8.0,
        reference="Flynn 1990; HuskinDB; Beetge et al., Int J Pharm 2000; 193:261",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="indomethacin", mw=357.79, logp=4.27,
        kp_cm_h=6.3e-4, jss_ug_cm2_h=0.63,
        lag_time_h=6.0,
        reference="Flynn 1990; Potts & Guy 1992; Cheruvu et al. 2022",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="ketoprofen", mw=254.28, logp=3.12,
        kp_cm_h=3.4e-3, jss_ug_cm2_h=3.4,
        lag_time_h=2.0,
        reference="Flynn 1990; Potts & Guy 1992; HuskinDB",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="flurbiprofen", mw=244.26, logp=4.16,
        kp_cm_h=7.8e-3, jss_ug_cm2_h=7.8,
        lag_time_h=1.5,
        reference="Flynn 1990; Cheruvu et al. 2022; HuskinDB",
        year=1990, formulation="aqueous",
    ),

    # ── Steroids (Flynn 1990, Scheuplein 1969/1971, HuskinDB) ────────
    PermeationRecord(
        drug_name="corticosterone", mw=346.46, logp=1.94,
        kp_cm_h=5.0e-4, jss_ug_cm2_h=0.25,
        lag_time_h=8.0,
        reference="Flynn 1990; Scheuplein et al. 1969; HuskinDB",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="progesterone", mw=314.46, logp=3.87,
        kp_cm_h=9.3e-3, jss_ug_cm2_h=0.93,
        lag_time_h=4.5,
        reference="Flynn 1990; Scheuplein et al. 1969; Potts & Guy 1992",
        year=1990, formulation="aqueous (saturated)",
        notes="High logP steroid — one of the most permeable steroids in Flynn set",
    ),
    PermeationRecord(
        drug_name="aldosterone", mw=360.44, logp=1.08,
        kp_cm_h=3.0e-4, jss_ug_cm2_h=0.15,
        lag_time_h=10.0,
        reference="Flynn 1990; Scheuplein et al. 1969",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="dexamethasone", mw=392.46, logp=1.83,
        kp_cm_h=4.0e-4, jss_ug_cm2_h=0.08,
        lag_time_h=8.0,
        reference="Flynn 1990; HuskinDB; Cheruvu et al. 2022",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="cortisone", mw=360.44, logp=1.47,
        kp_cm_h=3.0e-4,
        lag_time_h=9.0,
        reference="Flynn 1990; Scheuplein & Blank 1971",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="hydroxypregnenolone", mw=332.48, logp=2.44,
        kp_cm_h=9.1e-4,
        reference="Flynn 1990; Scheuplein et al. 1969",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="pregnenolone", mw=316.48, logp=4.22,
        kp_cm_h=6.0e-3,
        reference="Flynn 1990; Scheuplein et al. 1969",
        year=1990, formulation="aqueous (saturated)",
    ),

    # ── Transdermal drug targets (various published) ─────────────────
    PermeationRecord(
        drug_name="nitroglycerin", mw=227.09, logp=1.62,
        kp_cm_h=2.8e-3, jss_ug_cm2_h=14.0,
        lag_time_h=0.8,
        reference="Flynn 1990; Potts & Guy 1992; Tojo et al., J Pharm Sci 1987",
        year=1987, formulation="aqueous",
        notes="Transdermal patch prototype compound",
    ),
    PermeationRecord(
        drug_name="scopolamine", mw=303.35, logp=0.98,
        kp_cm_h=1.4e-3, jss_ug_cm2_h=0.70,
        lag_time_h=4.0,
        reference="Flynn 1990; Potts & Guy 1992; Chandrasekaran et al., AIChE Symp 1978",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="clonidine", mw=230.09, logp=1.59,
        kp_cm_h=1.3e-2, jss_ug_cm2_h=6.5,
        lag_time_h=1.0,
        reference="Potts & Guy 1992; Tojo et al. 1987; Cheruvu et al. 2022",
        year=1992, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="oxybutynin", mw=357.49, logp=4.30,
        kp_cm_h=8.5e-4, jss_ug_cm2_h=0.43,
        lag_time_h=5.0,
        reference="Cheruvu et al. 2022; Nicoli et al., Eur J Pharm Biopharm 2005; 61:56",
        year=2005, formulation="aqueous",
    ),

    # ── Vitamins and lipophilic compounds ────────────────────────────
    PermeationRecord(
        drug_name="minoxidil", mw=209.25, logp=-1.68,
        kp_cm_h=2.7e-4, jss_ug_cm2_h=0.27,
        lag_time_h=6.0,
        reference="Potts & Guy 1992; Chiang et al., J Pharm Sci 1989; 78:390",
        year=1989, formulation="aqueous",
        notes="Polar compound — logP negative despite topical use",
    ),
    PermeationRecord(
        drug_name="retinol", mw=286.45, logp=5.68,
        kp_cm_h=3.2e-3,
        reference="Pham et al., J Control Release 2016; 232:175; HuskinDB",
        year=2016, formulation="aqueous / ethanol cosolvent",
        notes="Highly lipophilic — aqueous solubility limits flux",
    ),
    PermeationRecord(
        drug_name="tocopherol", mw=430.71, logp=10.51,
        kp_cm_h=4.5e-5,
        reference="Nada et al., Int J Cosmet Sci 2011; 33:165; HuskinDB",
        year=2011, formulation="aqueous / solubilised",
        notes="Ultra-lipophilic — extremely poor aqueous flux",
    ),

    # ── Sugars and hydrophilics (Flynn 1990) ─────────────────────────
    PermeationRecord(
        drug_name="urea", mw=60.06, logp=-2.11,
        kp_cm_h=4.0e-4,
        reference="Flynn 1990; Scheuplein & Blank 1971",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="mannitol", mw=182.17, logp=-3.10,
        kp_cm_h=1.5e-5,
        reference="Flynn 1990; Scheuplein & Blank 1971",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="glucose", mw=180.16, logp=-3.24,
        kp_cm_h=6.0e-6,
        reference="Flynn 1990; Scheuplein & Blank 1971",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="sucrose", mw=342.30, logp=-3.70,
        kp_cm_h=1.3e-6,
        reference="Flynn 1990; Scheuplein & Blank 1971",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="thiourea", mw=76.12, logp=-0.95,
        kp_cm_h=2.3e-4,
        reference="Flynn 1990",
        year=1990, formulation="aqueous",
    ),

    # ── Volatile organics (Flynn 1990, HuskinDB) ────────────────────
    PermeationRecord(
        drug_name="benzene", mw=78.11, logp=2.13,
        kp_cm_h=1.5e-2,
        reference="Flynn 1990; Scheuplein & Blank 1971",
        year=1990, formulation="aqueous (saturated)",
    ),
    PermeationRecord(
        drug_name="toluene", mw=92.14, logp=2.73,
        kp_cm_h=1.0e-1,
        reference="Flynn 1990; Scheuplein & Blank 1971",
        year=1990, formulation="aqueous (saturated)",
    ),
    PermeationRecord(
        drug_name="chloroform", mw=119.38, logp=1.97,
        kp_cm_h=1.2e-2,
        reference="Flynn 1990; Scheuplein & Blank 1971",
        year=1990, formulation="aqueous (saturated)",
    ),
    PermeationRecord(
        drug_name="diethyl ether", mw=74.12, logp=0.89,
        kp_cm_h=4.0e-3,
        reference="Flynn 1990; Scheuplein 1971",
        year=1990, formulation="aqueous (saturated)",
    ),
    PermeationRecord(
        drug_name="styrene", mw=104.15, logp=2.95,
        kp_cm_h=9.0e-2,
        reference="Flynn 1990; HuskinDB",
        year=1990, formulation="aqueous (saturated)",
    ),

    # ── Alkaloids and miscellaneous drugs (Flynn / HuskinDB / various)
    PermeationRecord(
        drug_name="codeine", mw=299.36, logp=1.14,
        kp_cm_h=1.5e-3, jss_ug_cm2_h=1.5,
        lag_time_h=4.0,
        reference="Flynn 1990; Potts & Guy 1992; Roy & Flynn, Pharm Res 1989",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="morphine", mw=285.34, logp=0.89,
        kp_cm_h=8.0e-4, jss_ug_cm2_h=0.40,
        lag_time_h=5.0,
        reference="Flynn 1990; Roy & Flynn, Pharm Res 1989; 6:825",
        year=1989, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="atropine", mw=289.37, logp=1.83,
        kp_cm_h=2.4e-3,
        reference="Flynn 1990",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="ouabain", mw=584.65, logp=-1.70,
        kp_cm_h=1.6e-6,
        reference="Flynn 1990; Scheuplein & Blank 1971",
        year=1990, formulation="aqueous",
        notes="Large polar glycoside — extremely low permeability",
    ),
    PermeationRecord(
        drug_name="digitoxin", mw=764.94, logp=1.86,
        kp_cm_h=4.0e-6,
        reference="Flynn 1990; Scheuplein & Blank 1971",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="antipyrine", mw=188.23, logp=0.38,
        kp_cm_h=5.3e-4,
        reference="Flynn 1990; SkinPiX (Champmartin et al. 2024)",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="aminopyrine", mw=231.29, logp=1.00,
        kp_cm_h=6.0e-4,
        reference="Flynn 1990; SkinPiX (Champmartin et al. 2024)",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="barbital", mw=184.19, logp=0.65,
        kp_cm_h=2.0e-4,
        reference="Flynn 1990; Potts & Guy 1992",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="theophylline", mw=180.16, logp=-0.02,
        kp_cm_h=1.3e-3,
        reference="Flynn 1990; Potts & Guy 1992; HuskinDB",
        year=1990, formulation="aqueous",
    ),

    # ── Parabens (preservatives, well-studied homologous series) ─────
    PermeationRecord(
        drug_name="methyl paraben", mw=152.15, logp=1.96,
        kp_cm_h=7.0e-3,
        reference="Flynn 1990; Dal Pozzo & Pastori, Int J Pharm 1996; 138:63",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="ethyl paraben", mw=166.17, logp=2.47,
        kp_cm_h=1.3e-2,
        reference="Flynn 1990; Dal Pozzo & Pastori 1996",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="propyl paraben", mw=180.20, logp=3.04,
        kp_cm_h=2.7e-2,
        reference="Flynn 1990; HuskinDB; Dal Pozzo & Pastori 1996",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="butyl paraben", mw=194.23, logp=3.57,
        kp_cm_h=4.0e-2,
        reference="Flynn 1990; HuskinDB",
        year=1990, formulation="aqueous",
    ),

    # ── Additional HuskinDB / SkinPiX / Cheruvu compounds ───────────
    PermeationRecord(
        drug_name="acyclovir", mw=225.20, logp=-1.56,
        kp_cm_h=1.2e-5, jss_ug_cm2_h=0.012,
        lag_time_h=12.0,
        reference="HuskinDB; Parry et al., J Control Release 1992; 21:169",
        year=1992, formulation="aqueous",
        notes="Very hydrophilic antiviral — poor skin permeation",
    ),
    PermeationRecord(
        drug_name="metronidazole", mw=171.15, logp=-0.02,
        kp_cm_h=3.0e-4,
        reference="HuskinDB; Cheruvu et al. 2022; Patel et al., AAPS PharmSciTech 2011",
        year=2011, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="diazepam", mw=284.74, logp=2.82,
        kp_cm_h=4.0e-3, jss_ug_cm2_h=2.0,
        lag_time_h=2.0,
        reference="Flynn 1990; HuskinDB; Potts & Guy 1992",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="acetylsalicylic acid", mw=180.16, logp=1.19,
        kp_cm_h=3.6e-4,
        reference="Flynn 1990",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="piroxicam", mw=331.35, logp=3.06,
        kp_cm_h=2.8e-4,
        reference="SkinPiX 2024; Beetge et al. 2000",
        year=2024, formulation="aqueous",
        notes="SkinPiX updated value (slightly lower than Flynn)",
    ),
    PermeationRecord(
        drug_name="triamcinolone acetonide", mw=434.50, logp=2.53,
        kp_cm_h=3.2e-4, jss_ug_cm2_h=0.16,
        lag_time_h=8.0,
        reference="Cheruvu et al. 2022; HuskinDB",
        year=2022, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="fluocinolone acetonide", mw=452.49, logp=2.48,
        kp_cm_h=2.8e-4,
        reference="Cheruvu et al. 2022; HuskinDB",
        year=2022, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="clobetasol propionate", mw=466.97, logp=3.50,
        kp_cm_h=5.0e-4,
        reference="Cheruvu et al. 2022",
        year=2022, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="diflorasone diacetate", mw=494.52, logp=3.28,
        kp_cm_h=3.0e-4,
        reference="HuskinDB; Cheruvu et al. 2022",
        year=2022, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="econazole", mw=381.68, logp=5.46,
        kp_cm_h=2.3e-3,
        reference="HuskinDB; SkinPiX 2024; Krishnaiah et al., J Drug Deliv Sci Tech 2005",
        year=2005, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="ketoconazole", mw=531.43, logp=4.35,
        kp_cm_h=6.0e-5,
        reference="HuskinDB; SkinPiX 2024",
        year=2020, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="miconazole", mw=416.13, logp=6.10,
        kp_cm_h=1.2e-3,
        reference="HuskinDB; SkinPiX 2024",
        year=2020, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="terbinafine", mw=291.43, logp=5.58,
        kp_cm_h=4.5e-3,
        reference="SkinPiX 2024; Alberti et al., Eur J Pharm Biopharm 2001; 51:131",
        year=2001, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="nifedipine", mw=346.34, logp=2.20,
        kp_cm_h=1.4e-3,
        reference="Flynn 1990; Potts & Guy 1992",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="sufentanil", mw=386.55, logp=3.95,
        kp_cm_h=2.1e-3,
        reference="Roy & Flynn, Pharm Res 1990; 7:842; HuskinDB",
        year=1990, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="buprenorphine", mw=467.64, logp=4.98,
        kp_cm_h=1.8e-3,
        reference="HuskinDB; Cheruvu et al. 2022; Panchagnula et al., Pharm Res 2005",
        year=2005, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="naloxone", mw=327.37, logp=2.09,
        kp_cm_h=6.5e-4,
        reference="HuskinDB; Cheruvu et al. 2022",
        year=2022, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="granisetron", mw=312.41, logp=2.65,
        kp_cm_h=3.5e-3,
        reference="HuskinDB; Cheruvu et al. 2022",
        year=2022, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="ondansetron", mw=293.37, logp=2.40,
        kp_cm_h=2.8e-3,
        reference="HuskinDB; Cheruvu et al. 2022",
        year=2022, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="donepezil", mw=379.49, logp=4.11,
        kp_cm_h=1.5e-3,
        reference="SkinPiX 2024; Choi et al., Eur J Pharm Biopharm 2012; 82:320",
        year=2012, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="rivastigmine", mw=250.34, logp=2.30,
        kp_cm_h=5.0e-3,
        reference="SkinPiX 2024; Hadgraft & Lane, Int J Pharm 2006; 307:167",
        year=2006, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="rotigotine", mw=315.47, logp=4.70,
        kp_cm_h=6.0e-3,
        reference="SkinPiX 2024; Cheruvu et al. 2022",
        year=2022, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="ketamine", mw=237.73, logp=3.12,
        kp_cm_h=6.0e-3, jss_ug_cm2_h=6.0,
        lag_time_h=1.5,
        reference="SkinPiX 2024; Inoue et al., Int J Pharm 2019; 571:118766",
        year=2019, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="tamoxifen", mw=371.51, logp=6.30,
        kp_cm_h=5.0e-4,
        reference="SkinPiX 2024; Cheruvu et al. 2022",
        year=2022, formulation="aqueous",
        notes="Very lipophilic — limited aqueous flux",
    ),
    PermeationRecord(
        drug_name="imiquimod", mw=240.30, logp=2.70,
        kp_cm_h=8.0e-4,
        reference="SkinPiX 2024; HuskinDB",
        year=2020, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="adapalene", mw=412.52, logp=8.04,
        kp_cm_h=2.0e-5,
        reference="SkinPiX 2024; Choi et al., Skin Pharmacol Physiol 2014",
        year=2014, formulation="aqueous / cosolvent",
        notes="Ultra-lipophilic retinoid — minimal aqueous permeation",
    ),
    PermeationRecord(
        drug_name="diclofenac sodium", mw=318.13, logp=1.13,
        kp_cm_h=1.0e-3, jss_ug_cm2_h=1.0,
        lag_time_h=3.5,
        reference="SkinPiX 2024; Singh & Roberts, J Pharmacol Exp Ther 1994",
        year=1994, formulation="aqueous (ionised form)",
        notes="Ionised form — lower Kp than un-ionised diclofenac",
    ),
    PermeationRecord(
        drug_name="etoricoxib", mw=358.84, logp=3.14,
        kp_cm_h=1.8e-3,
        reference="SkinPiX 2024",
        year=2024, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="methotrexate", mw=454.44, logp=-1.85,
        kp_cm_h=5.0e-6,
        reference="SkinPiX 2024; Cheruvu et al. 2022",
        year=2022, formulation="aqueous",
        notes="Large polar anti-metabolite — extremely poor permeation",
    ),
    PermeationRecord(
        drug_name="coumarin", mw=146.14, logp=1.39,
        kp_cm_h=4.2e-3,
        reference="HuskinDB v1.01; Potts & Guy 1992",
        year=1992, formulation="aqueous",
    ),
    PermeationRecord(
        drug_name="etodolac", mw=287.36, logp=3.40,
        kp_cm_h=3.0e-3,
        reference="HuskinDB v1.01; SkinPiX 2024",
        year=2020, formulation="aqueous",
    ),
]


class HuskinDBScraper:
    """Fetches and integrates data from HuskinDB, SkinPiX, and Cheruvu
    databases into the virtual skin permeation framework.

    If live download fails, a large manually curated dataset compiled from
    published compilations (Flynn 1990, Potts & Guy 1992, HuskinDB 2020,
    SkinPiX 2024, Cheruvu et al. 2022) is used instead.
    """

    def __init__(self) -> None:
        self._raw_rows: List[Dict[str, Any]] = []
        self._records: List[PermeationRecord] = []
        self._download_succeeded = False

    # ─── public API ──────────────────────────────────────────────────

    def fetch_database(self, timeout: int = 30) -> bool:
        """Try to download data from HuskinDB / SkinPiX / Cheruvu.

        Attempts, in order:
          1. HuskinDB data page (CSV export)
          2. Recherche Data Gouv API for SkinPiX / HuskinDB subsets
          3. Mendeley dataset (Cheruvu et al.)

        Returns True if at least one source succeeded.
        """
        import urllib.request

        success = False

        for label, url in [
            ("HuskinDB-data", HUSKINDB_URLS["data_page"]),
            ("SkinPiX-huskin-subset", SKINPIX_URLS["data_gouv_huskin"]),
            ("SkinPiX-skinpix-subset", SKINPIX_URLS["data_gouv_skinpix"]),
        ]:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "VirtualSkin/1.0"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                    rows = self._parse_tsv_response(raw)
                    if rows:
                        logger.info(f"Fetched {len(rows)} rows from {label}")
                        self._raw_rows.extend(rows)
                        success = True
            except (URLError, TimeoutError, OSError) as exc:
                logger.warning(f"Could not fetch {label}: {exc}")
            except Exception as exc:
                logger.warning(f"Unexpected error fetching {label}: {exc}")

        self._download_succeeded = success
        if not success:
            logger.info(
                "Live download failed — will use manually curated dataset "
                f"({len(CURATED_PERMEATION_RECORDS)} records)"
            )
        return success

    def parse_to_records(self) -> List[PermeationRecord]:
        """Convert fetched + curated data to PermeationRecord objects.

        Always starts with the curated dataset, then adds any successfully
        downloaded records on top (deduplicating by drug name + reference).
        """
        self._records = list(CURATED_PERMEATION_RECORDS)

        if self._download_succeeded and self._raw_rows:
            downloaded = self._convert_raw_rows(self._raw_rows)
            if downloaded:
                logger.info(f"Converted {len(downloaded)} records from download")
                existing_keys = {(r.drug_name.lower(), r.reference) for r in self._records}
                for rec in downloaded:
                    key = (rec.drug_name.lower(), rec.reference)
                    if key not in existing_keys:
                        self._records.append(rec)
                        existing_keys.add(key)
            else:
                logger.info("Downloaded data could not be parsed — using curated only")

        return self._records

    def merge_with_literature_db(
        self,
        db: Optional[IVPTLiteratureDB] = None,
    ) -> IVPTLiteratureDB:
        """Merge parsed records into an IVPTLiteratureDB instance.

        New drugs get their own key; existing drugs get additional records
        appended (duplicates filtered by reference string).
        """
        if db is None:
            db = IVPTLiteratureDB()

        if not self._records:
            self.parse_to_records()

        added, skipped = 0, 0
        for rec in self._records:
            key = rec.drug_name.lower().replace(" ", "_").replace("-", "_")
            existing = db.records.get(key, [])

            existing_refs = {r.reference for r in existing}
            if rec.reference in existing_refs:
                skipped += 1
                continue

            if key not in db.records:
                db.records[key] = []
            db.records[key].append(rec)
            added += 1

        logger.info(
            f"Merge complete: {added} records added, {skipped} duplicates skipped. "
            f"DB now has {db.n_records} records across {len(db.drug_names)} drugs."
        )
        return db

    # ─── convenience ─────────────────────────────────────────────────

    @property
    def n_curated(self) -> int:
        return len(CURATED_PERMEATION_RECORDS)

    @property
    def n_fetched(self) -> int:
        return len(self._raw_rows)

    @property
    def download_succeeded(self) -> bool:
        return self._download_succeeded

    def get_curated_records(self) -> List[PermeationRecord]:
        """Return the built-in curated dataset without network access."""
        return list(CURATED_PERMEATION_RECORDS)

    def summary(self) -> str:
        lines = [
            f"HuskinDBScraper — {self.n_curated} curated records built-in",
            f"  Download attempted: {'yes' if self._download_succeeded else 'no'}",
            f"  Raw rows fetched:   {self.n_fetched}",
            f"  Parsed records:     {len(self._records)}",
        ]
        drug_set = {r.drug_name for r in (self._records or CURATED_PERMEATION_RECORDS)}
        lines.append(f"  Unique drugs:       {len(drug_set)}")
        return "\n".join(lines)

    # ─── internal helpers ────────────────────────────────────────────

    @staticmethod
    def _parse_tsv_response(raw: str) -> List[Dict[str, str]]:
        """Best-effort parse of a TSV / CSV response from data repositories."""
        rows: List[Dict[str, str]] = []
        try:
            dialect = csv.Sniffer().sniff(raw[:2048])
            reader = csv.DictReader(io.StringIO(raw), dialect=dialect)
            for row in reader:
                rows.append(dict(row))
        except csv.Error:
            for sep in ["\t", ",", ";"]:
                try:
                    reader = csv.DictReader(io.StringIO(raw), delimiter=sep)
                    for row in reader:
                        if len(row) > 2:
                            rows.append(dict(row))
                    if rows:
                        break
                except csv.Error:
                    continue
        return rows

    @staticmethod
    def _safe_float(val: Any) -> Optional[float]:
        if val is None:
            return None
        try:
            v = float(str(val).strip())
            if v != v:  # NaN check
                return None
            return v
        except (ValueError, TypeError):
            return None

    def _convert_raw_rows(self, rows: List[Dict[str, str]]) -> List[PermeationRecord]:
        """Map downloaded table columns to PermeationRecord fields.

        Handles both HuskinDB and SkinPiX column naming conventions.
        """
        records: List[PermeationRecord] = []

        name_cols = ["Compound", "compound", "Name", "name", "COMPOUND", "substance"]
        mw_cols = ["MW", "mw", "MolecularWeight", "molecular_weight", "MW (g/mol)"]
        logp_cols = ["LogP", "logP", "logp", "LogKow", "logKow", "log P"]
        kp_cols = ["Kp", "kp", "Kp (cm/h)", "kp_cm_h", "Permeability coefficient"]
        jss_cols = ["Jss", "jss", "Jmax", "Jss (µg/cm²/h)", "jss_ug_cm2_h", "Flux"]
        lag_cols = ["tlag", "Lag time", "lag_time_h", "tlag (h)"]

        def _find(row: Dict[str, str], candidates: List[str]) -> Optional[str]:
            for c in candidates:
                if c in row and row[c]:
                    return row[c].strip()
            return None

        for row in rows:
            name = _find(row, name_cols)
            mw = self._safe_float(_find(row, mw_cols))
            logp = self._safe_float(_find(row, logp_cols))
            kp = self._safe_float(_find(row, kp_cols))
            jss = self._safe_float(_find(row, jss_cols))
            lag = self._safe_float(_find(row, lag_cols))

            if not name or mw is None or logp is None:
                continue

            records.append(PermeationRecord(
                drug_name=name.lower(),
                mw=mw,
                logp=logp,
                kp_cm_h=kp,
                jss_ug_cm2_h=jss,
                lag_time_h=lag,
                reference=_find(row, ["Reference", "reference", "Source", "Publication"]) or "HuskinDB/SkinPiX",
                year=int(self._safe_float(_find(row, ["Year", "year"])) or 2020),
                formulation=_find(row, ["Vehicle", "vehicle", "Donor type"]) or "aqueous",
                skin_source=_find(row, ["Species", "species", "Skin"]) or "human",
                site=_find(row, ["Site", "site", "Skin source site"]) or "unknown",
            ))

        return records


def build_expanded_literature_db(try_download: bool = True) -> IVPTLiteratureDB:
    """One-call convenience: build an IVPTLiteratureDB with all available data.

    Combines the original curated database with HuskinDB / SkinPiX / Cheruvu
    records (downloaded or from the built-in curated set).
    """
    scraper = HuskinDBScraper()
    if try_download:
        scraper.fetch_database()
    scraper.parse_to_records()
    db = scraper.merge_with_literature_db()
    logger.info(db.summary())
    return db
