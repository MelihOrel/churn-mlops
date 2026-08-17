"""Model loading and batch/single prediction.

A tiny module-level cache means the API loads the joblib artifact once per
process. The pipeline carries its own preprocessing, so callers pass raw
feature values exactly as they appear in the source data.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from .config import Config, load_config
from .schema import FeatureSpec


class ModelNotFoundError(FileNotFoundError):
    """Raised when no serialized model artifact exists yet."""


def _default_model_path(cfg: Config) -> Path:
    return Path(cfg.get("model.registry_dir", "models")) / cfg.get(
        "model.model_filename", "churn_model.joblib"
    )


@lru_cache(maxsize=1)
def load_model(model_path: str | None = None):
    cfg = load_config()
    path = Path(model_path) if model_path else _default_model_path(cfg)
    if not path.exists():
        raise ModelNotFoundError(
            f"No model at '{path}'. Train one first: python -m churn.train"
        )
    return joblib.load(path)


def load_spec() -> FeatureSpec:
    """Return the feature contract stored with the trained model."""
    bundle = load_model()
    return FeatureSpec.from_dict(bundle["spec"])


def predict_proba(records: list[dict], cfg: Config | None = None) -> list[float]:
    """Return churn probability for each record dict."""
    bundle = load_model()
    pipeline = bundle["pipeline"]
    features = bundle["features"]

    frame = pd.DataFrame(records)
    missing = set(features) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing feature(s) in request: {sorted(missing)}")

    proba = pipeline.predict_proba(frame[features])[:, 1]
    return [float(p) for p in proba]


def get_threshold() -> float:
    """The operating point chosen at training time and stored with the model."""
    bundle = load_model()
    stored = bundle.get("threshold")
    if stored is not None:
        return float(stored)
    configured = load_config().get("serving.decision_threshold", 0.5)
    return 0.5 if configured == "auto" else float(configured)


def predict(records: list[dict], threshold: float | None = None) -> list[dict]:
    cfg = load_config()
    threshold = get_threshold() if threshold is None else float(threshold)
    probs = predict_proba(records, cfg)
    return [
        {
            "churn_probability": round(p, 4),
            "churn_prediction": bool(p >= threshold),
            "risk_band": "high" if p >= 0.66 else "medium" if p >= 0.33 else "low",
        }
        for p in probs
    ]
