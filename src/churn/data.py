"""Data loading and validation.

Validation is a first-class MLOps concern: bad data silently poisons models.
We check schema, types, ranges and target sanity *before* anything trains,
and fail loudly with a clear message when an expectation is violated.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Config


class DataValidationError(ValueError):
    """Raised when incoming data violates the expected schema or ranges."""


def load_raw(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset '{path}' not found. Run: python scripts/generate_data.py"
        )
    return pd.read_csv(path)


def validate(df: pd.DataFrame, cfg: Config, *, require_target: bool = True) -> pd.DataFrame:
    """Validate ``df`` against the config schema; return a cleaned copy.

    Cleaning here is intentionally minimal and deterministic (coerce numerics,
    drop exact-duplicate rows). Anything that looks structurally wrong raises
    ``DataValidationError`` rather than being silently patched.
    """
    df = df.copy()
    numeric = cfg.get("features.numeric", [])
    categorical = cfg.get("features.categorical", [])
    target = cfg.get("data.target", "Churn")

    expected = set(numeric) | set(categorical)
    if require_target:
        expected.add(target)
    missing = expected - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing required columns: {sorted(missing)}")

    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    nan_counts = df[numeric].isna().sum()
    bad = nan_counts[nan_counts > 0]
    if len(bad) > 0:
        # Numeric columns that failed to parse are a data-quality red flag.
        raise DataValidationError(
            f"Non-numeric values found in numeric columns: {bad.to_dict()}"
        )

    if (df[numeric] < 0).any().any():
        raise DataValidationError("Negative values found in numeric feature columns.")

    if require_target:
        labels = set(df[target].dropna().unique())
        if not labels.issubset({"Yes", "No"}):
            raise DataValidationError(
                f"Target '{target}' must be Yes/No, got: {sorted(labels)}"
            )

    if cfg.get("data.drop_duplicates", False):
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            print(f"[validate] dropped {dropped} duplicate row(s)")

    return df


def split_features_target(df: pd.DataFrame, cfg: Config):
    """Return (X, y) where y is a 0/1 int series (1 == churned)."""
    target = cfg.get("data.target", "Churn")
    X = df[cfg.feature_columns].copy()
    y = (df[target] == "Yes").astype(int)
    return X, y
