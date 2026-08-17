"""The feature contract that travels with the model.

At training time we capture the exact feature names, dtypes and the observed
category values, then serialize that spec alongside the fitted pipeline. The
serving layer builds its request validation *from the spec*, not from a
hand-written schema — so the API can never drift out of sync with the model it
is serving, and swapping datasets requires no API code changes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


@dataclass
class FeatureSpec:
    """Feature contract: names, order, and allowed values per categorical."""

    numeric: list[str] = field(default_factory=list)
    categorical: dict[str, list[str]] = field(default_factory=dict)
    numeric_ranges: dict[str, list[float]] = field(default_factory=dict)
    source: str = "unknown"
    target: str = "Churn"

    @property
    def columns(self) -> list[str]:
        return list(self.numeric) + list(self.categorical)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureSpec:
        return cls(
            numeric=list(data.get("numeric", [])),
            categorical={k: list(v) for k, v in data.get("categorical", {}).items()},
            numeric_ranges={k: list(v) for k, v in data.get("numeric_ranges", {}).items()},
            source=data.get("source", "unknown"),
            target=data.get("target", "Churn"),
        )

    def example(self) -> dict[str, Any]:
        """A syntactically valid example payload, for API docs."""
        payload: dict[str, Any] = {}
        for col in self.numeric:
            lo, hi = self.numeric_ranges.get(col, [0.0, 1.0])
            payload[col] = round((lo + hi) / 2, 2)
        for col, values in self.categorical.items():
            payload[col] = values[0] if values else ""
        return payload


def build_spec(df: pd.DataFrame, cfg) -> FeatureSpec:
    """Derive the spec from the training frame, driven by the config lists."""
    numeric = list(cfg.get("features.numeric", []))
    categorical = list(cfg.get("features.categorical", []))

    return FeatureSpec(
        numeric=numeric,
        categorical={
            col: sorted(str(v) for v in df[col].dropna().unique()) for col in categorical
        },
        numeric_ranges={
            col: [float(df[col].min()), float(df[col].max())] for col in numeric
        },
        source=cfg.source,
        target=cfg.get("data.target", "Churn"),
    )
