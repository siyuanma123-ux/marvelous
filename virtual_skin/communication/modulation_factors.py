"""
Map consensus crosstalk modules to transport parameter modulation factors.

This is the key bridge: cell communication → equation control variables.
Each module activity adjusts one or more transport parameters through
physically interpretable modulation rules.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .consensus_modules import ConsensusCrosstalkBuilder, SKIN_MODULE_DEFINITIONS


class CommunicationModulationMapper:
    """Convert consensus module activities into transport-parameter modulation factors."""

    def __init__(
        self,
        module_defs: Optional[Dict] = None,
        sensitivity: float = 1.0,
    ) -> None:
        self.defs = module_defs or SKIN_MODULE_DEFINITIONS
        self.sensitivity = sensitivity

    def compute_modulation_factors(
        self,
        builder: ConsensusCrosstalkBuilder,
    ) -> Dict[str, float]:
        """Return multiplicative modulation factors for each open transport parameter.

        A factor > 1 means the module increases the parameter;
        a factor < 1 means the module decreases it.
        Neutral (no modulation) = 1.0.
        """
        factors: Dict[str, float] = {}
        for mod_name, mod_def in self.defs.items():
            target = mod_def["target_param"]
            direction = mod_def["direction"]
            module = builder.modules.get(mod_name)
            if module is None:
                continue

            activity = module.activity * self.sensitivity

            if direction == "positive":
                factor = 1.0 + activity
            elif direction == "negative":
                factor = 1.0 / (1.0 + activity)
            else:
                factor = 1.0

            # Accumulate (multiplicative) if multiple modules target same param
            factors[target] = factors.get(target, 1.0) * factor

        return factors

    def apply_to_params(
        self,
        base_params: Dict[str, float],
        builder: ConsensusCrosstalkBuilder,
    ) -> Dict[str, float]:
        """Apply modulation factors to base transport parameters."""
        factors = self.compute_modulation_factors(builder)
        modulated = dict(base_params)
        for param, factor in factors.items():
            if param in modulated:
                modulated[param] = modulated[param] * factor
        return modulated

    def explain_modulation(
        self, builder: ConsensusCrosstalkBuilder
    ) -> List[Dict[str, str]]:
        """Human-readable explanation of each active modulation."""
        explanations = []
        for mod_name, mod_def in self.defs.items():
            module = builder.modules.get(mod_name)
            if module is None or module.activity < 0.05:
                continue
            direction = "increases" if mod_def["direction"] == "positive" else "decreases"
            explanations.append({
                "module": mod_name,
                "target_param": mod_def["target_param"],
                "direction": direction,
                "activity": f"{module.activity:.3f}",
                "confidence": f"{module.consensus_confidence:.2f}",
                "explanation": (
                    f"{mod_def['description']}: activity = {module.activity:.3f} "
                    f"→ {direction} {mod_def['target_param']}"
                ),
            })
        return explanations
