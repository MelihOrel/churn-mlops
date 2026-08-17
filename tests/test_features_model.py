"""Tests for preprocessing, metrics and drift."""
import numpy as np
import pandas as pd

from churn.evaluate import compute_metrics
from churn.features import build_preprocessor
from churn.monitoring import compute_drift


def test_preprocessor_transforms_to_numeric_matrix(sample_df, cfg):
    pre = build_preprocessor(cfg)
    X = sample_df[cfg.feature_columns]
    matrix = pre.fit_transform(X)
    assert matrix.shape[0] == len(sample_df)
    assert np.issubdtype(np.asarray(matrix).dtype, np.number)


def test_preprocessor_handles_unseen_category(sample_df, cfg):
    pre = build_preprocessor(cfg)
    pre.fit(sample_df[cfg.feature_columns])
    unseen = sample_df.iloc[[0]].copy()
    unseen.loc[:, "PaymentMethod"] = "Crypto"  # never seen at fit time
    # handle_unknown="ignore" must not raise.
    assert pre.transform(unseen[cfg.feature_columns]).shape[0] == 1


def test_metrics_are_in_valid_range():
    y = np.array([0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.4, 0.35, 0.8, 0.9])
    m = compute_metrics(y, proba)
    for key in ("roc_auc", "pr_auc", "precision", "recall", "f1"):
        assert 0.0 <= m[key] <= 1.0


def test_drift_detects_shift(cfg):
    rng = np.random.default_rng(0)
    n = 500
    base = {
        "tenure": rng.integers(0, 72, n),
        "MonthlyCharges": rng.normal(70, 20, n),
        "TotalCharges": rng.normal(1500, 500, n),
        "NumSupportTickets": rng.poisson(1.5, n),
        "Contract": rng.choice(["Month-to-month", "Two year"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic"], n),
        "PaymentMethod": rng.choice(["Electronic check", "Credit card"], n),
        "HasPremiumSupport": rng.choice(["Yes", "No"], n),
    }
    ref = pd.DataFrame(base)
    cur = ref.copy()
    cur["MonthlyCharges"] = cur["MonthlyCharges"] + 60  # strong shift
    report = compute_drift(ref, cur, cfg)
    assert report["dataset_drift"] is True
    assert "MonthlyCharges" in report["drifted_features"]


def test_drift_stable_when_same_distribution(cfg):
    rng = np.random.default_rng(1)
    n = 500
    data = {
        "tenure": rng.integers(0, 72, n),
        "MonthlyCharges": rng.normal(70, 20, n),
        "TotalCharges": rng.normal(1500, 500, n),
        "NumSupportTickets": rng.poisson(1.5, n),
        "Contract": rng.choice(["Month-to-month", "Two year"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic"], n),
        "PaymentMethod": rng.choice(["Electronic check", "Credit card"], n),
        "HasPremiumSupport": rng.choice(["Yes", "No"], n),
    }
    ref = pd.DataFrame(data)
    cur = pd.DataFrame(data)
    report = compute_drift(ref, cur, cfg)
    assert report["dataset_drift"] is False
