"""Shared config / vocab / seeding helpers. Local files only, no network."""
from __future__ import annotations

import json
import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
SEED = 42


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_path(rel: str) -> Path:
    return REPO_ROOT / rel


def get_num_channels() -> int:
    """NUM_CHANNELS is read from config.yaml (7 default, 6 = z-ablation); never hard-coded."""
    n = int(load_config()["num_channels"])
    if n not in (6, 7):
        raise ValueError(f"config.yaml num_channels must be 6 or 7, got {n}")
    return n


def seed_everything(seed: int = SEED) -> None:
    """Seed python, numpy and torch (if installed). Call once at every entry point."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # features/interpolator do not need torch
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seeding (pass as worker_init_fn)."""
    s = SEED + worker_id
    random.seed(s)
    np.random.seed(s)


def load_vocab(path: str | Path | None = None) -> list[str]:
    """Read the gloss list. Accepts a plain list or {"glosses": [...]}. num_classes = len(vocab)."""
    p = Path(path) if path else repo_path(load_config()["paths"]["vocab"])
    if not p.is_file():
        raise FileNotFoundError(f"vocab file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    glosses = raw["glosses"] if isinstance(raw, dict) else raw
    if not isinstance(glosses, list) or not glosses:
        raise ValueError(f"{p}: expected a non-empty list of glosses")
    glosses = [str(g) for g in glosses]
    if any(g != g.upper() for g in glosses):
        raise ValueError(f"{p}: all glosses must be UPPERCASE")
    if len(set(glosses)) != len(glosses):
        raise ValueError(f"{p}: duplicate glosses")
    return glosses


def vocab_is_proxy(path: str | Path | None = None) -> bool:
    p = Path(path) if path else repo_path(load_config()["paths"]["vocab"])
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return bool(isinstance(raw, dict) and raw.get("_proxy", False))
