"""
End-to-end data integration: public datasets → Virtual Skin pipeline inputs.

This module glues scRNA-seq, spatial transcriptomics, and IVPT data into
the format expected by VirtualSkinSolver.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class PublicDataIntegrator:
    """Orchestrates download, preprocessing, and formatting of public datasets
    into the inputs expected by the VirtualSkinSolver pipeline.

    Usage:
        integrator = PublicDataIntegrator(output_dir="data/public")
        scrna, spatial, ivpt = integrator.prepare_all()
    """

    def __init__(self, output_dir: str = "data/public"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare_scrna(
        self,
        dataset_id: str = "GSE147424",
        force: bool = False,
    ) -> str:
        """Download/create scRNA-seq data. Returns h5ad path."""
        from .scrna_datasets import download_skin_scrna
        return download_skin_scrna(dataset_id, str(self.output_dir), force)

    def prepare_spatial(
        self,
        dataset_id: str = "visium_human_skin_ffpe",
        force: bool = False,
    ) -> str:
        """Download/create spatial data. Returns h5ad path."""
        from .spatial_datasets import download_skin_spatial
        return download_skin_spatial(dataset_id, str(self.output_dir), force)

    def prepare_ivpt(
        self,
        drugs: Optional[List[str]] = None,
        conditions: Optional[List[str]] = None,
        n_replicates: int = 3,
    ) -> Dict[str, Any]:
        """Generate IVPT validation curves from literature.

        Returns a dict with 'curves', 'summary_df', 'db' keys.
        """
        import pandas as pd
        from .ivpt_literature import IVPTLiteratureDB

        db = IVPTLiteratureDB()
        logger.info(f"IVPT Literature DB:\n{db.summary()}")

        curves = db.generate_multi_drug_validation_set(drugs, conditions, n_replicates)

        rows = []
        for c in curves:
            rows.append({
                "drug": c["drug"],
                "condition": c["condition"],
                "replicate": c["replicate"],
                "jss_target": c["jss_target"],
                "lag_time_target": c["lag_time_target"],
                "total_permeation_24h": c["cumulative_ug_cm2"][-1] if len(c["cumulative_ug_cm2"]) > 0 else 0,
            })
        summary_df = pd.DataFrame(rows)

        csv_path = self.output_dir / "ivpt_literature_summary.csv"
        summary_df.to_csv(csv_path, index=False)
        logger.info(f"IVPT summary saved to {csv_path}")

        # Also save full database
        db_df = db.to_dataframe()
        db_df.to_csv(self.output_dir / "ivpt_database_full.csv", index=False)

        return {"curves": curves, "summary_df": summary_df, "db": db}

    def prepare_all(
        self,
        scrna_id: str = "GSE147424",
        spatial_id: str = "visium_human_skin_ffpe",
        ivpt_drugs: Optional[List[str]] = None,
        force: bool = False,
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Prepare all three data types in one call.

        Returns: (scrna_h5ad_path, spatial_h5ad_path, ivpt_data_dict)
        """
        logger.info("=" * 60)
        logger.info("Preparing public datasets for Virtual Skin pipeline")
        logger.info("=" * 60)

        scrna_path = self.prepare_scrna(scrna_id, force)
        spatial_path = self.prepare_spatial(spatial_id, force)
        ivpt_data = self.prepare_ivpt(ivpt_drugs)

        logger.info("\n" + "=" * 60)
        logger.info("All datasets prepared:")
        logger.info(f"  scRNA-seq: {scrna_path}")
        logger.info(f"  Spatial:   {spatial_path}")
        logger.info(f"  IVPT:      {len(ivpt_data['curves'])} curves from literature")
        logger.info("=" * 60)

        return scrna_path, spatial_path, ivpt_data

    def format_for_solver(
        self,
        scrna_path: str,
        spatial_path: str,
        ivpt_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Convert prepared data into the exact format expected by VirtualSkinSolver.

        Returns a dict that can be unpacked as kwargs to solver methods.
        """
        import anndata as ad
        from ..skin_sample import SkinSample, SkinSite, SkinCondition
        from ..ivpt_data import FranzDiffusionData, IVPTExperiment

        adata_sc = ad.read_h5ad(scrna_path)
        adata_sp = ad.read_h5ad(spatial_path)

        _site_map = {
            "forearm": SkinSite.FOREARM, "abdomen": SkinSite.ABDOMEN,
            "face": SkinSite.FACE, "back": SkinSite.BACK,
        }

        # Build SkinSample objects from scRNA metadata
        samples = []
        if "condition" in adata_sc.obs.columns:
            for cond in adata_sc.obs["condition"].unique():
                mask = adata_sc.obs["condition"] == cond
                site_str = adata_sc.obs.loc[mask, "site"].iloc[0] if "site" in adata_sc.obs.columns else "forearm"
                site_enum = _site_map.get(str(site_str).lower(), SkinSite.FOREARM)

                condition_enum = SkinCondition.HEALTHY
                if "atopic" in str(cond).lower():
                    condition_enum = SkinCondition.ATOPIC_DERMATITIS
                elif "psoriasis" in str(cond).lower():
                    condition_enum = SkinCondition.PSORIASIS

                samples.append(SkinSample(
                    donor_id=f"donor_{cond}",
                    site=site_enum,
                    condition=condition_enum,
                ))

        # Build IVPTExperiment objects (grouped by drug × formulation)
        from collections import defaultdict
        grouped: Dict[str, List] = defaultdict(list)
        for curve in ivpt_data["curves"]:
            fd = FranzDiffusionData(
                sample_id=f"{curve['drug']}_{curve['condition']}_{curve['replicate']}",
                drug_name=curve["drug"],
                formulation="aqueous",
                time_h=curve["time_h"],
                cumulative_permeation=curve["cumulative_ug_cm2"],
            )
            grouped[curve["drug"]].append(fd)

        ivpt_experiments = []
        for drug_name, runs in grouped.items():
            exp = IVPTExperiment(drug_name=drug_name, formulation="aqueous")
            for run in runs:
                exp.add_run(run)
            ivpt_experiments.append(exp)

        return {
            "adata_scrna": adata_sc,
            "adata_spatial": adata_sp,
            "samples": samples,
            "ivpt_experiments": ivpt_experiments,
        }

    def generate_demo_report(self) -> str:
        """Generate a summary report of available public data."""
        from .scrna_datasets import AVAILABLE_SCRNA_DATASETS
        from .spatial_datasets import AVAILABLE_SPATIAL_DATASETS
        from .ivpt_literature import IVPTLiteratureDB

        lines = ["# Public Data for Virtual Skin Model", ""]

        lines.append("## 1. scRNA-seq Datasets")
        for k, v in AVAILABLE_SCRNA_DATASETS.items():
            lines.append(f"### {k}: {v['title']}")
            lines.append(f"- Publication: {v.get('publication', 'N/A')}")
            lines.append(f"- Conditions: {v.get('conditions', [])}")
            lines.append(f"- Cells: {v.get('n_cells', 'N/A')}")
            lines.append(f"- Notes: {v.get('notes', '')}")
            lines.append("")

        lines.append("## 2. Spatial Transcriptomics")
        for k, v in AVAILABLE_SPATIAL_DATASETS.items():
            lines.append(f"### {k}: {v['title']}")
            lines.append(f"- Platform: {v.get('platform', 'N/A')}")
            lines.append(f"- Free download: {v.get('free_download', 'check website')}")
            lines.append(f"- Notes: {v.get('notes', '')}")
            lines.append("")

        lines.append("## 3. IVPT Literature Database")
        db = IVPTLiteratureDB()
        lines.append(db.summary())
        lines.append("")

        lines.append("## 4. Recommended Workflow")
        lines.append("1. Start with GSE147424 (scRNA, healthy+AD) + synthetic spatial")
        lines.append("2. Use IVPT literature DB for initial calibration")
        lines.append("3. Download 10x Visium skin for real spatial validation")
        lines.append("4. Expand to GSE162183 for psoriasis comparison")
        lines.append("5. Replace with your private data for final publication")

        return "\n".join(lines)
