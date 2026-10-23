"""Load the YAML config and apply CLI overrides."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    """Read a config file, then apply `key.sub=value` overrides (values parsed as YAML)."""
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    if os.environ.get("VPR_DATA_ROOT"):
        cfg["data_root"] = os.environ["VPR_DATA_ROOT"]
    return cfg


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the repo root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def method_config(cfg: dict, method: str) -> dict:
    if method not in cfg.get("methods", {}):
        raise KeyError(f"No config section methods.{method} in config")
    return copy.deepcopy(cfg["methods"][method] or {})
