"""Public dataset acquisition, preprocessing, and integration for the Virtual Skin system."""

from .ivpt_literature import IVPTLiteratureDB, DRUG_PERMEATION_DATABASE
from .huskindb_scraper import (
    HuskinDBScraper,
    CURATED_PERMEATION_RECORDS,
    build_expanded_literature_db,
)

# Lazy (may pull scanpy): scrna_datasets, spatial_datasets, data_integration
# Import directly when needed: from .scrna_datasets import download_skin_scrna
