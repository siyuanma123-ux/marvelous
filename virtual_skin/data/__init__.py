"""Data management for multi-modal paired skin samples."""

from .drug_properties import DrugDescriptor, DrugLibrary
from .ivpt_data import IVPTExperiment, FranzDiffusionData
from .skin_sample import SkinSample, SkinSampleCollection

# Lazy: omics_loader (scanpy) — import directly when needed:
#   from virtual_skin.data.omics_loader import load_scrna, load_spatial
