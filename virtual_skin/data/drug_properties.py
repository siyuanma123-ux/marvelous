"""
Drug physicochemical descriptor management.

Descriptors entering the state-modulation layer:
  molecular_weight, logP, pKa, charge, binding_affinity, …
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class DrugDescriptor:
    """Physicochemical profile of a single drug."""

    name: str
    molecular_weight: float          # Da
    logP: float                      # octanol-water partition coefficient
    pKa: Optional[float] = None
    charge_at_pH7: float = 0.0
    hydrogen_bond_donors: int = 0
    hydrogen_bond_acceptors: int = 0
    polar_surface_area: float = 0.0  # Å²
    binding_affinity: float = 0.0    # proxy for ECM/protein binding tendency
    solubility_mg_mL: Optional[float] = None

    metadata: Dict = field(default_factory=dict)

    def to_vector(self) -> np.ndarray:
        """Return normalizable feature vector for model input."""
        return np.array(
            [
                self.molecular_weight,
                self.logP,
                self.pKa if self.pKa is not None else 7.0,
                self.charge_at_pH7,
                self.hydrogen_bond_donors,
                self.hydrogen_bond_acceptors,
                self.polar_surface_area,
                self.binding_affinity,
            ],
            dtype=np.float32,
        )

    @property
    def descriptor_names(self) -> List[str]:
        return [
            "molecular_weight",
            "logP",
            "pKa",
            "charge_at_pH7",
            "hbd",
            "hba",
            "PSA",
            "binding_affinity",
        ]


class DrugLibrary:
    """Registry of drug descriptor profiles."""

    def __init__(self) -> None:
        self._drugs: Dict[str, DrugDescriptor] = {}

    def add(self, drug: DrugDescriptor) -> None:
        self._drugs[drug.name] = drug

    def get(self, name: str) -> DrugDescriptor:
        return self._drugs[name]

    def list_drugs(self) -> List[str]:
        return sorted(self._drugs.keys())

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for d in self._drugs.values():
            row = {"name": d.name}
            row.update(dict(zip(d.descriptor_names, d.to_vector())))
            rows.append(row)
        return pd.DataFrame(rows)

    def get_descriptor_matrix(self, drug_names: List[str]) -> np.ndarray:
        """Return (n_drugs, n_descriptors) matrix."""
        return np.stack([self._drugs[n].to_vector() for n in drug_names])

    @classmethod
    def from_csv(cls, path: str) -> "DrugLibrary":
        df = pd.read_csv(path)
        lib = cls()
        for _, row in df.iterrows():
            lib.add(
                DrugDescriptor(
                    name=row["name"],
                    molecular_weight=row["molecular_weight"],
                    logP=row["logP"],
                    pKa=row.get("pKa"),
                    charge_at_pH7=row.get("charge_at_pH7", 0.0),
                    hydrogen_bond_donors=int(row.get("hbd", 0)),
                    hydrogen_bond_acceptors=int(row.get("hba", 0)),
                    polar_surface_area=row.get("PSA", 0.0),
                    binding_affinity=row.get("binding_affinity", 0.0),
                )
            )
        return lib

    # ---- Built-in reference drugs ----
    @classmethod
    def default_library(cls) -> "DrugLibrary":
        lib = cls()
        lib.add(DrugDescriptor("caffeine", 194.2, -0.07, pKa=10.4, solubility_mg_mL=20.0))
        lib.add(DrugDescriptor("lidocaine", 234.3, 2.44, pKa=7.9, charge_at_pH7=0.5, solubility_mg_mL=10.0))
        lib.add(DrugDescriptor("diclofenac", 296.1, 4.51, pKa=4.15, charge_at_pH7=-0.9, solubility_mg_mL=1.0))
        lib.add(DrugDescriptor("hydrocortisone", 362.5, 1.61, pKa=12.6, solubility_mg_mL=0.5))
        lib.add(DrugDescriptor("testosterone", 288.4, 3.32, pKa=15.0, solubility_mg_mL=1.0))
        lib.add(DrugDescriptor("nicotine", 162.2, 1.17, pKa=3.12, charge_at_pH7=0.9, solubility_mg_mL=10.0))
        lib.add(DrugDescriptor("fentanyl", 336.5, 4.05, pKa=8.99, charge_at_pH7=0.6, solubility_mg_mL=0.2))
        lib.add(DrugDescriptor("ibuprofen", 206.3, 3.97, pKa=4.91, charge_at_pH7=-0.9, solubility_mg_mL=0.04))
        lib.add(DrugDescriptor("estradiol", 272.4, 4.01, solubility_mg_mL=0.003))
        lib.add(DrugDescriptor("progesterone", 314.5, 3.87, solubility_mg_mL=0.01))
        lib.add(DrugDescriptor("naproxen", 230.3, 3.18, pKa=4.15, charge_at_pH7=-0.9, solubility_mg_mL=0.02))
        lib.add(DrugDescriptor("salicylic_acid", 138.1, 2.26, pKa=2.97, charge_at_pH7=-1.0, solubility_mg_mL=2.0))
        lib.add(DrugDescriptor("minoxidil", 209.3, 1.24, solubility_mg_mL=2.2))
        lib.add(DrugDescriptor("piroxicam", 331.3, 3.06, pKa=6.3, charge_at_pH7=-0.3, solubility_mg_mL=0.02))
        lib.add(DrugDescriptor("ketoprofen", 254.3, 3.12, pKa=4.45, charge_at_pH7=-0.9, solubility_mg_mL=0.05))
        return lib
