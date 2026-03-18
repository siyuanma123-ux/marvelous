"""
Multi-modal paired skin sample management.

Each donor / site combination produces a paired sample set:
  - IVPT tissue block  →  Franz diffusion data
  - scRNA-seq / snRNA-seq block  →  cell state atlas
  - Spatial transcriptomics block  →  spatial niche map
  - Histology / IF block  →  layer morphometry and marker validation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import anndata as ad
import numpy as np
import pandas as pd


class SkinSite(Enum):
    FOREARM = "forearm"
    ABDOMEN = "abdomen"
    FACE = "face"
    SOLE = "sole"
    BACK = "back"
    SCALP = "scalp"


class SkinCondition(Enum):
    HEALTHY = "healthy"
    ATOPIC_DERMATITIS = "atopic_dermatitis"
    PSORIASIS = "psoriasis"
    FIBROTIC = "fibrotic"
    WOUND = "wound"
    AGED = "aged"


@dataclass
class SkinSample:
    """A single multi-modal paired skin sample."""

    donor_id: str
    site: SkinSite
    condition: SkinCondition = SkinCondition.HEALTHY
    age: Optional[int] = None
    sex: Optional[str] = None

    # Omics data (populated after loading)
    adata_sc: Optional[ad.AnnData] = field(default=None, repr=False)
    adata_spatial: Optional[ad.AnnData] = field(default=None, repr=False)

    # Histology metadata
    sc_thickness_um: Optional[float] = None
    ve_thickness_um: Optional[float] = None
    dermis_thickness_um: Optional[float] = None
    follicle_density: Optional[float] = None

    # Derived state axes (computed by atlas module)
    state_vector: Optional[np.ndarray] = field(default=None, repr=False)

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def sample_id(self) -> str:
        return f"{self.donor_id}_{self.site.value}_{self.condition.value}"

    def has_paired_omics(self) -> bool:
        return self.adata_sc is not None and self.adata_spatial is not None

    def get_layer_thicknesses(self) -> Dict[str, float]:
        """Return measured layer thicknesses (µm), falling back to defaults."""
        return {
            "stratum_corneum": self.sc_thickness_um or 15.0,
            "viable_epidermis": self.ve_thickness_um or 80.0,
            "dermis": self.dermis_thickness_um or 1200.0,
        }


class SkinSampleCollection:
    """Registry of all donor–site paired skin samples."""

    def __init__(self) -> None:
        self._samples: Dict[str, SkinSample] = {}

    def add(self, sample: SkinSample) -> None:
        self._samples[sample.sample_id] = sample

    def get(self, sample_id: str) -> SkinSample:
        return self._samples[sample_id]

    def filter(
        self,
        site: Optional[SkinSite] = None,
        condition: Optional[SkinCondition] = None,
        donor_ids: Optional[List[str]] = None,
    ) -> List[SkinSample]:
        results = list(self._samples.values())
        if site is not None:
            results = [s for s in results if s.site == site]
        if condition is not None:
            results = [s for s in results if s.condition == condition]
        if donor_ids is not None:
            results = [s for s in results if s.donor_id in donor_ids]
        return results

    def list_donors(self) -> List[str]:
        return sorted({s.donor_id for s in self._samples.values()})

    def summary(self) -> pd.DataFrame:
        rows = []
        for s in self._samples.values():
            rows.append(
                {
                    "sample_id": s.sample_id,
                    "donor": s.donor_id,
                    "site": s.site.value,
                    "condition": s.condition.value,
                    "has_sc": s.adata_sc is not None,
                    "has_spatial": s.adata_spatial is not None,
                    "has_state": s.state_vector is not None,
                }
            )
        return pd.DataFrame(rows)

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self):
        return iter(self._samples.values())
