"""Skin state atlas: computable state space from multi-modal omics.

Lightweight entry: from .state_vector import TissueStateVector
Full omics encoders: from .state_space import SkinStateSpace (requires scanpy)
"""

from .state_vector import TissueStateVector, STATE_AXIS_NAMES

__all__ = ["TissueStateVector", "STATE_AXIS_NAMES"]
