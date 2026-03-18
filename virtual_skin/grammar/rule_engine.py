"""
Rule execution engine: evaluate grammar rules against tissue state
and apply to transport parameters with conflict resolution.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .hypothesis_grammar import HypothesisRule, SkinBehaviorGrammar
from ..atlas.state_space import TissueStateVector


class RuleEngine:
    """Evaluates grammar rules and produces modulated transport parameters."""

    def __init__(self, grammar: Optional[SkinBehaviorGrammar] = None) -> None:
        self.grammar = grammar or SkinBehaviorGrammar.default_skin_grammar()

    def evaluate(
        self,
        tissue_state: TissueStateVector,
        base_params: Dict[str, float],
    ) -> Tuple[Dict[str, float], List[Dict]]:
        """Apply grammar rules to base parameters.

        Returns:
            modulated_params: updated transport parameters
            audit_log: list of rule activations for transparency
        """
        state_dict = {
            name: val
            for name, val in zip(
                TissueStateVector.axis_names(), tissue_state.to_array()
            )
        }

        triggered = self.grammar.evaluate_all(state_dict)
        audit_log = []
        params = dict(base_params)

        # Sort by confidence (higher confidence rules take priority)
        triggered.sort(key=lambda r: r.confidence, reverse=True)

        for rule in triggered:
            if rule.effect_param in params:
                old_val = params[rule.effect_param]
                new_val = rule.compute_effect(old_val)
                params[rule.effect_param] = new_val
                audit_log.append({
                    "rule_id": rule.rule_id,
                    "rule_name": rule.name,
                    "condition_met": True,
                    "param": rule.effect_param,
                    "old_value": old_val,
                    "new_value": new_val,
                    "confidence": rule.confidence,
                    "natural_language": rule.to_natural_language(),
                })

        return params, audit_log

    def counterfactual(
        self,
        tissue_state: TissueStateVector,
        base_params: Dict[str, float],
        override_rules: Optional[List[str]] = None,
        disable_rules: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Run with specific rules overridden or disabled for counterfactual analysis."""
        backup = {}
        if disable_rules:
            for rid in disable_rules:
                if rid in self.grammar.rules:
                    backup[rid] = self.grammar.rules.pop(rid)

        params, _ = self.evaluate(tissue_state, base_params)

        # Restore disabled rules
        for rid, rule in backup.items():
            self.grammar.rules[rid] = rule

        return params

    def sensitivity_to_rules(
        self,
        tissue_state: TissueStateVector,
        base_params: Dict[str, float],
    ) -> Dict[str, Dict[str, float]]:
        """For each rule, compute the parameter change if ONLY that rule fires."""
        results = {}
        for rid, rule in self.grammar.rules.items():
            state_dict = {
                name: val
                for name, val in zip(
                    TissueStateVector.axis_names(), tissue_state.to_array()
                )
            }
            if rule.evaluate_condition(state_dict) and rule.effect_param in base_params:
                old = base_params[rule.effect_param]
                new = rule.compute_effect(old)
                results[rid] = {
                    "param": rule.effect_param,
                    "base": old,
                    "modulated": new,
                    "fold_change": new / old if old != 0 else float("inf"),
                    "confidence": rule.confidence,
                }
        return results
