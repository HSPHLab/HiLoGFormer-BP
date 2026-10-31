"""
Config system.

An experiment YAML lists the files it builds on under `defaults` (applied in order),
then overrides only what changes. Command-line `--set a.b=value` overrides come last.
The fully resolved config is saved in every run folder and inside every checkpoint.

    # configs/experiments/hicaf_bp.yaml
    defaults: [base.yaml, models/hicaf_bp.yaml]
    experiment: {name: hicaf_bp}
    train: {lr: 3.0e-4}
"""
import copy
from pathlib import Path

import yaml

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


def deep_update(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    merged = {}
    for d in cfg.pop("defaults", []):
        merged = deep_update(merged, _load(CONFIG_ROOT / d))
    return deep_update(merged, cfg)


def _parse_value(s: str):
    """'3e-4' -> float, 'true' -> bool, '[1,2]' -> list, otherwise string (YAML rules)."""
    return yaml.safe_load(s)


def apply_overrides(cfg: dict, overrides) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got: {item}")
        key, val = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(val)
    return cfg


def load_config(path, overrides=None) -> dict:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_ROOT / path
    cfg = _load(path)
    cfg = apply_overrides(cfg, overrides)
    cfg.setdefault("experiment", {})["config_file"] = str(path)
    return cfg


def save_config(cfg: dict, path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
