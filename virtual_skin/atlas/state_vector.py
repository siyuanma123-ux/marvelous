"""
Lightweight tissue state vector — no heavy dependencies (scanpy, etc.).
Used by transport model and Streamlit app. Full state_space.py adds omics encoders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


STATE_AXIS_NAMES: List[str] = [
    "barrier_integrity",
    "inflammatory_load",
    "ecm_remodeling",
    "vascularization",
    "appendage_openness",
]


@dataclass
class TissueStateVector:
    """Low-dimensional tissue state that enters the transport equation."""

    barrier_integrity: float = 1.0
    inflammatory_load: float = 0.0
    ecm_remodeling: float = 0.0
    vascularization: float = 0.5
    appendage_openness: float = 0.0

    layer_state_raw: Optional[Dict[str, np.ndarray]] = field(default=None, repr=False)
    niche_state_raw: Optional[np.ndarray] = field(default=None, repr=False)
    cell_state_raw: Optional[np.ndarray] = field(default=None, repr=False)

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.barrier_integrity,
                self.inflammatory_load,
                self.ecm_remodeling,
                self.vascularization,
                self.appendage_openness,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def axis_names() -> List[str]:
        return STATE_AXIS_NAMES
