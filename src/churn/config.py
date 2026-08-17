"""Typed configuration loading with multi-dataset resolution.

A single YAML file drives every stage of the pipeline. The config supports
several data sources; ``data.source`` picks one and its block is merged up into
``data`` and ``features`` at load time. Downstream stages therefore read the
same keys regardless of which dataset is active — no branching on source.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = os.environ.get("CHURN_CONFIG", "config/config.yaml")


class ConfigError(ValueError):
    """Raised when the configuration is internally inconsistent."""


@dataclass
class Config:
    """Thin wrapper over the parsed YAML config with dotted-path access."""

    raw: dict[str, Any] = field(default_factory=dict)

    def get(self, path: str, default: Any = None) -> Any:
        """Fetch a nested value using a dotted path, e.g. ``get('model.type')``."""
        node: Any = self.raw
        for key in path.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    @property
    def source(self) -> str:
        return self.get("data.source", "synthetic")

    @property
    def feature_columns(self) -> list[str]:
        return list(self.get("features.numeric", [])) + list(
            self.get("features.categorical", [])
        )


def _resolve_active_dataset(data: dict[str, Any]) -> dict[str, Any]:
    """Merge the selected dataset block into ``data`` and ``features``."""
    data = copy.deepcopy(data)
    source = data.get("data", {}).get("source", "synthetic")
    datasets = data.get("datasets", {})

    if source not in datasets:
        raise ConfigError(
            f"data.source='{source}' has no matching entry under 'datasets'. "
            f"Available: {sorted(datasets)}"
        )

    block = copy.deepcopy(datasets[source])
    features = block.pop("features", None)
    if not features:
        raise ConfigError(f"Dataset '{source}' defines no 'features' block.")

    data["features"] = features
    data["data"] = {**data.get("data", {}), **block}
    return data


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from ``path`` (defaults to ``config/config.yaml``)."""
    cfg_path = Path(path or DEFAULT_CONFIG_PATH)
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config file not found at '{cfg_path}'. Set CHURN_CONFIG or pass a path."
        )
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    # Allow an env override so CI can switch datasets without editing the file.
    env_source = os.environ.get("CHURN_DATA_SOURCE")
    if env_source:
        data.setdefault("data", {})["source"] = env_source

    return Config(raw=_resolve_active_dataset(data))
