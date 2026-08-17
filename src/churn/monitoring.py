"""Data drift monitoring via Population Stability Index (PSI).

In production a model decays when incoming data drifts away from what it was
trained on. PSI is a lightweight, dependency-free way to quantify that shift
per feature; anything above the configured threshold is flagged so a retrain
can be triggered. The stage writes a JSON report suitable for CI or a cron job.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, load_config
from .data import load_raw, validate


def _psi_numeric(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """PSI for a continuous feature using quantile bins from the reference."""
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if len(edges) < 3:  # not enough spread to bin meaningfully
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e_counts = np.histogram(expected, bins=edges)[0] / len(expected)
    a_counts = np.histogram(actual, bins=edges)[0] / len(actual)
    return _psi_from_dists(e_counts, a_counts)


def _psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    categories = expected.value_counts(normalize=True)
    actual_dist = actual.value_counts(normalize=True)
    e = categories.values
    a = actual_dist.reindex(categories.index).fillna(0.0).values
    return _psi_from_dists(e, a)


def _psi_from_dists(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-6) -> float:
    expected = np.clip(expected, eps, None)
    actual = np.clip(actual, eps, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame, cfg: Config) -> dict:
    numeric = cfg.get("features.numeric", [])
    categorical = cfg.get("features.categorical", [])
    threshold = cfg.get("monitoring.drift_threshold", 0.15)

    per_feature = {}
    for col in numeric:
        per_feature[col] = _psi_numeric(reference[col].values, current[col].values)
    for col in categorical:
        per_feature[col] = _psi_categorical(reference[col], current[col])

    drifted = {c: v for c, v in per_feature.items() if v > threshold}
    return {
        "threshold": threshold,
        "psi_by_feature": {c: round(v, 4) for c, v in per_feature.items()},
        "drifted_features": sorted(drifted, key=drifted.get, reverse=True),
        "n_drifted": len(drifted),
        "dataset_drift": len(drifted) > 0,
    }


def run_drift_report(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    ref = validate(load_raw(cfg.get("data.reference_path")), cfg, require_target=False)
    cur = validate(load_raw(cfg.get("data.current_path")), cfg, require_target=False)

    report = compute_drift(ref, cur, cfg)

    report_dir = Path(cfg.get("monitoring.report_dir", "reports"))
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / "drift_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    status = "DRIFT DETECTED" if report["dataset_drift"] else "stable"
    print(f"[monitoring] {status} — {report['n_drifted']} feature(s) drifted")
    print(f"[monitoring] report -> {out}")
    return report


if __name__ == "__main__":
    run_drift_report()
