"""Closed-loop validation framework for the virtual skin system."""

from .metrics import ValidationMetrics
from .ivpt_validation import IVPTValidator
from .perturbation import PerturbationValidator
from .blind_test import BlindExtrapolationTest
