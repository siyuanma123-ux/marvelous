"""
Skin Behavior Hypothesis Grammar.

Inspired by Cell 2025: human-interpretable grammar that writes cell-behaviour
hypotheses as executable rules, enabling virtual "thought experiments" and
testable predictions.

Each rule takes the form:
  IF <condition on tissue state / communication module>
  THEN <effect on transport parameter>
  WITH <confidence, evidence source, experimental test>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class EvidenceSource(Enum):
    LITERATURE = "literature"
    DATA_DRIVEN = "data_driven"
    COMBINED = "combined"
    HYPOTHETICAL = "hypothetical"


class ExperimentalTest(Enum):
    EX_VIVO_IVPT = "ex_vivo_ivpt"
    AUGMENTED_HSE = "augmented_hse"
    SPHEROID = "spheroid"
    CHIP = "chip"
    IF_STAINING = "if_staining"
    OMICS = "omics"


@dataclass
class HypothesisRule:
    """A single mechanistic hypothesis expressed as an executable rule."""

    rule_id: str
    name: str
    description: str

    # Condition
    condition_state: str      # state axis or module name
    condition_operator: str   # "increases", "decreases", "above_threshold", "below_threshold"

    # Effect
    effect_param: str         # transport parameter affected
    effect_direction: str     # "increase", "decrease"
    effect_magnitude: str     # "weak", "moderate", "strong"

    # Optional condition threshold
    condition_threshold: Optional[float] = None

    # Meta
    confidence: float = 0.5
    evidence: EvidenceSource = EvidenceSource.COMBINED
    evidence_refs: List[str] = field(default_factory=list)
    experimental_test: List[ExperimentalTest] = field(default_factory=list)
    falsifiable: bool = True

    def evaluate_condition(self, state: Dict[str, float]) -> bool:
        """Check whether the condition is met given current tissue state."""
        val = state.get(self.condition_state, 0.0)
        if self.condition_operator == "increases" and val > 0.3:
            return True
        if self.condition_operator == "decreases" and val < 0.3:
            return True
        if self.condition_operator == "above_threshold":
            return val > (self.condition_threshold or 0.5)
        if self.condition_operator == "below_threshold":
            return val < (self.condition_threshold or 0.5)
        return False

    def compute_effect(self, base_value: float) -> float:
        """Apply the rule effect to a base parameter value."""
        magnitude_map = {"weak": 0.2, "moderate": 0.5, "strong": 1.0}
        mag = magnitude_map.get(self.effect_magnitude, 0.3)

        if self.effect_direction == "increase":
            return base_value * (1.0 + mag)
        elif self.effect_direction == "decrease":
            return base_value / (1.0 + mag)
        return base_value

    def to_natural_language(self) -> str:
        return (
            f"RULE [{self.rule_id}]: When {self.condition_state} {self.condition_operator}"
            f"{f' (threshold={self.condition_threshold})' if self.condition_threshold else ''}, "
            f"{self.effect_param} will {self.effect_direction} ({self.effect_magnitude}). "
            f"[confidence={self.confidence:.2f}, evidence={self.evidence.value}]"
        )


class SkinBehaviorGrammar:
    """Collection of hypothesis rules forming the grammar layer."""

    def __init__(self) -> None:
        self.rules: Dict[str, HypothesisRule] = {}

    def add_rule(self, rule: HypothesisRule) -> None:
        self.rules[rule.rule_id] = rule

    def remove_rule(self, rule_id: str) -> None:
        self.rules.pop(rule_id, None)

    def evaluate_all(
        self, state: Dict[str, float]
    ) -> List[HypothesisRule]:
        """Return all rules whose conditions are met by the current state."""
        return [r for r in self.rules.values() if r.evaluate_condition(state)]

    def apply_rules(
        self, state: Dict[str, float], base_params: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply all triggered rules to base parameters."""
        params = dict(base_params)
        for rule in self.evaluate_all(state):
            if rule.effect_param in params:
                params[rule.effect_param] = rule.compute_effect(
                    params[rule.effect_param]
                )
        return params

    def list_rules(self) -> List[str]:
        return [r.to_natural_language() for r in self.rules.values()]

    @classmethod
    def default_skin_grammar(cls) -> "SkinBehaviorGrammar":
        """Built-in rules grounded in literature and atlas evidence."""
        grammar = cls()

        grammar.add_rule(HypothesisRule(
            rule_id="R1",
            name="Inflammatory myofibroblast → ECM retention",
            description=(
                "When inflammatory myofibroblast-related state rises, ECM remodeling "
                "increases dermal retention (k_bind_dermis)."
            ),
            condition_state="ecm_remodeling",
            condition_operator="above_threshold",
            condition_threshold=0.4,
            effect_param="k_bind_dermis",
            effect_direction="increase",
            effect_magnitude="moderate",
            confidence=0.8,
            evidence=EvidenceSource.COMBINED,
            evidence_refs=[
                "skin_fibroblast_atlas (Nat Commun)",
                "HSE single-cell (Cell Rep)",
            ],
            experimental_test=[
                ExperimentalTest.AUGMENTED_HSE,
                ExperimentalTest.EX_VIVO_IVPT,
            ],
        ))

        grammar.add_rule(HypothesisRule(
            rule_id="R2",
            name="Macrophage–endothelial module → vascular clearance",
            description=(
                "When macrophage–endothelial crosstalk strengthens, dermal vascular "
                "clearance (k_clear_vasc) and exposure redistribution increase."
            ),
            condition_state="vascularization",
            condition_operator="above_threshold",
            condition_threshold=0.5,
            effect_param="k_clear_vasc",
            effect_direction="increase",
            effect_magnitude="strong",
            confidence=0.75,
            evidence=EvidenceSource.COMBINED,
            evidence_refs=[
                "Prenatal skin atlas (Nature 2024)",
                "Macrophage–vascular remodeling",
            ],
            experimental_test=[
                ExperimentalTest.AUGMENTED_HSE,
                ExperimentalTest.CHIP,
            ],
        ))

        grammar.add_rule(HypothesisRule(
            rule_id="R3",
            name="Keratinocyte EMT-like state → barrier disruption",
            description=(
                "When keratinocyte EMT-like programme is elevated, barrier integrity "
                "decreases, SC/VE effective permeability increases."
            ),
            condition_state="barrier_integrity",
            condition_operator="below_threshold",
            condition_threshold=0.4,
            effect_param="D_sc",
            effect_direction="increase",
            effect_magnitude="moderate",
            confidence=0.7,
            evidence=EvidenceSource.COMBINED,
            evidence_refs=["HSE single-cell (Cell Rep)"],
            experimental_test=[
                ExperimentalTest.AUGMENTED_HSE,
                ExperimentalTest.EX_VIVO_IVPT,
                ExperimentalTest.IF_STAINING,
            ],
        ))

        grammar.add_rule(HypothesisRule(
            rule_id="R4",
            name="Inflammatory load → effective permeability",
            description=(
                "High inflammatory state disrupts tight junctions and increases "
                "SC diffusivity and VE permeability."
            ),
            condition_state="inflammatory_load",
            condition_operator="above_threshold",
            condition_threshold=0.5,
            effect_param="D_sc",
            effect_direction="increase",
            effect_magnitude="strong",
            confidence=0.85,
            evidence=EvidenceSource.LITERATURE,
            evidence_refs=["Inflammatory skin barrier disruption (multiple)"],
            experimental_test=[
                ExperimentalTest.EX_VIVO_IVPT,
                ExperimentalTest.AUGMENTED_HSE,
            ],
        ))

        grammar.add_rule(HypothesisRule(
            rule_id="R5",
            name="Appendage niche activity → bypass pathway",
            description=(
                "Active follicular/glandular niche increases appendage bypass weight."
            ),
            condition_state="appendage_openness",
            condition_operator="above_threshold",
            condition_threshold=0.3,
            effect_param="w_appendage",
            effect_direction="increase",
            effect_magnitude="moderate",
            confidence=0.6,
            evidence=EvidenceSource.LITERATURE,
            experimental_test=[ExperimentalTest.EX_VIVO_IVPT],
        ))

        grammar.add_rule(HypothesisRule(
            rule_id="R6",
            name="High barrier integrity → reduced SC diffusivity",
            description=(
                "Intact, well-differentiated stratum corneum reduces effective D_sc."
            ),
            condition_state="barrier_integrity",
            condition_operator="above_threshold",
            condition_threshold=0.7,
            effect_param="D_sc",
            effect_direction="decrease",
            effect_magnitude="moderate",
            confidence=0.9,
            evidence=EvidenceSource.LITERATURE,
            experimental_test=[ExperimentalTest.EX_VIVO_IVPT],
        ))

        return grammar
