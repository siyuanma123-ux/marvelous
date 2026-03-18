"""Global configuration loader."""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import numpy as np


_CFG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
_DEFAULT_CFG = os.path.join(_CFG_DIR, "default_config.yaml")


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    path = path or _DEFAULT_CFG
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cfg: Dict[str, Any] | None = None) -> torch.device:
    if cfg is not None and cfg.get("project", {}).get("device") == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")
