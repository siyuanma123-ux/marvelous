# Omics-Constrained Multi-Scale Virtual Skin

> **A "structure–cell–communication–transport" integrated digital skin system for transdermal drug delivery prediction, drug screening, and target-action prediction.**

## Overview

This project implements an **omics-constrained virtual skin** that transforms the traditional view of skin as a passive layered barrier into a dynamic, multi-cellular system where tissue structure, cell states, spatial micro-environments, cell–cell communication, and drug physicochemical properties jointly determine drug transport behavior.

### Key innovations

1. **State-dependent transport parameters** — Diffusion coefficients, partition coefficients, and retention/clearance terms are not fixed constants but state-dependent variables modulated by tissue biology.

2. **Multi-tier state representation** — Layer state (SC, VE, dermis), niche state (perivascular, inflammatory, fibrotic, etc.), and cell state (keratinocyte, fibroblast, endothelial, macrophage sub-states) are encoded from multi-modal omics data.

3. **Cell communication as equation control variables** — Consensus crosstalk modules (barrier maintenance, inflammatory permeability, ECM retention, vascular clearance, appendage bypass) derived from CellChat, COMMOT, and FlowSig directly modulate PDE parameters.

4. **Human-interpretable hypothesis grammar** — Mechanistic rules expressed in natural language are compiled into executable modulation functions, enabling virtual "thought experiments" and falsifiable predictions.

5. **Closed-loop validation** — Ex vivo human skin IVPT data calibrates the model; augmented HSE, spheroid, and chip systems provide directed perturbation validation.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    VirtualSkinSolver                            │
├───────────────┬─────────────────┬───────────────┬──────────────┤
│  Atlas Module │   Communication │   Transport   │  Validation  │
│               │     Module      │    Module      │   Module     │
│  ┌──────────┐│  ┌────────────┐ │  ┌──────────┐ │  ┌─────────┐ │
│  │ GraphST  ││  │  CellChat  │ │  │ Layered  │ │  │  IVPT   │ │
│  │ wrapper  ││  │  (liana)   │ │  │ Fick PDE │ │  │ Validat.│ │
│  ├──────────┤│  ├────────────┤ │  ├──────────┤ │  ├─────────┤ │
│  │Nicheform.││  │   COMMOT   │ │  │  PINN    │ │  │Perturb. │ │
│  │ wrapper  ││  │  (Sp. OT)  │ │  │ solver   │ │  │ Valid.  │ │
│  ├──────────┤│  ├────────────┤ │  ├──────────┤ │  ├─────────┤ │
│  │  Layer / ││  │  FlowSig   │ │  │  State   │ │  │ Blind   │ │
│  │  Niche / ││  │ (dir.flow) │ │  │ modulat. │ │  │  test   │ │
│  │  Cell    ││  ├────────────┤ │  ├──────────┤ │  └─────────┘ │
│  │  State   ││  │ Consensus  │ │  │Bayesian  │ │              │
│  │ encoders ││  │  modules   │ │  │ calibr.  │ │              │
│  └──────────┘│  └────────────┘ │  └──────────┘ │              │
├───────────────┴─────────────────┴───────────────┤              │
│              Grammar / Rule Engine               │              │
│  ┌──────────┐ ┌───────────┐ ┌────────────────┐  │              │
│  │Hypothesis│ │   Rule    │ │   Virtual      │  │              │
│  │ Grammar  │ │  Engine   │ │  Experiment    │  │              │
│  └──────────┘ └───────────┘ └────────────────┘  │              │
└──────────────────────────────────────────────────┴──────────────┘
```

## Installation

```bash
pip install -e .
# Or install dependencies directly:
pip install -r requirements.txt
```

## Quick Start

```python
from virtual_skin.solver import VirtualSkinSolver
from virtual_skin.atlas.state_space import TissueStateVector
from virtual_skin.data.drug_properties import DrugLibrary

# Initialize solver
solver = VirtualSkinSolver()

# Define tissue state (from omics or manually)
tissue_state = TissueStateVector(
    barrier_integrity=0.8,
    inflammatory_load=0.1,
    ecm_remodeling=0.2,
    vascularization=0.5,
    appendage_openness=0.1,
)

# Get drug
drug = DrugLibrary.default_library().get("caffeine")

# Predict
prediction = solver.predict(tissue_state=tissue_state, drug=drug)
print(f"Jss = {prediction.steady_state_flux:.4f} µg/cm²/h")
print(f"Lag time = {prediction.lag_time:.2f} h")

# Virtual experiment
result = solver.virtual_experiment(drug, axis="inflammatory_load", target_value=0.8)
print(f"ΔJss with inflammation = {result['delta_flux']:.4f}")
```

## Module Reference

| Module | Description | Reference code used |
|--------|-------------|-------------------|
| `virtual_skin.atlas` | Multi-modal state space construction | GraphST, Nicheformer |
| `virtual_skin.communication` | Cell–cell communication inference | CellChat (via liana), COMMOT, FlowSig |
| `virtual_skin.transport` | Physics-informed transport model | deepxde (PINN architecture) |
| `virtual_skin.grammar` | Hypothesis grammar and rule engine | Cell 2025 grammar concept |
| `virtual_skin.validation` | Closed-loop validation framework | — |
| `virtual_skin.solver` | Integrated virtual skin solver | — |

## Run Scripts

```bash
# Full demo pipeline
python scripts/run_virtual_skin.py

# Build state atlas
python scripts/run_atlas_construction.py --sc_path data/sc.h5ad --st_path data/st.h5ad

# Transport predictions for all drugs
python scripts/run_transport_model.py --drug all --state 0.8,0.1,0.2,0.5,0.1

# Validation pipeline
python scripts/run_validation.py
```

## Project Structure

```
coding/
├── configs/default_config.yaml
├── virtual_skin/
│   ├── data/           # Sample management, IVPT, omics loaders, drug properties
│   ├── atlas/          # GraphST, Nicheformer wrappers; state encoders
│   ├── communication/  # CellChat, COMMOT, FlowSig; consensus modules
│   ├── transport/      # Layered PDE, PINN solver, state modulation, Bayesian
│   ├── grammar/        # Hypothesis grammar, rule engine, virtual experiments
│   ├── validation/     # IVPT, perturbation, blind testing, metrics
│   ├── solver/         # Integrated VirtualSkinSolver
│   └── visualization/  # Spatial, transport, and validation plots
├── scripts/            # Run scripts for each pipeline stage
└── notebooks/          # Jupyter notebooks (to be populated)
```

## Theoretical Foundation

- **AI Virtual Cell** (Cell 2024): Multi-scale, multi-modal simulation framework
- **Virtual Cell technical roadmap** (npj Digital Medicine 2025): Multi-omics + deep generative models + PINN + closed-loop validation
- **Human-interpretable grammar** (Cell 2025): Rule-driven, mechanism-auditable modeling
- **GraphST**: GNN + contrastive learning for spatial transcriptomics
- **Nicheformer**: Spatial-aware foundation model with technology/species tokens
- **CellChat**: Mass-action-based cell–cell communication probability
- **COMMOT**: Collective optimal transport for spatially-resolved CCC
- **FlowSig**: Directional intercellular information flow

## License

Research use only.
