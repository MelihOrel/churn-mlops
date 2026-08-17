"""Tests for cost-sensitive threshold selection and the feature contract."""
from __future__ import annotations

import numpy as np
import pandas as pd

from churn.evaluate import expected_value, tune_threshold
from churn.schema import FeatureSpec, build_spec


def test_expected_value_rewards_caught_churners():
    y_true = np.array([1, 1, 0, 0])
    catch_all = np.array([1, 1, 1, 1])
    catch_none = np.array([0, 0, 0, 0])

    kwargs = {"offer_cost": 50, "churn_loss": 500, "offer_success_rate": 0.3}
    # 2 TP * (0.3*500 - 50) = 2*100 = 200, minus 2 FP * 50 = 100 -> 100
    assert expected_value(y_true, catch_all, **kwargs) == 100.0
    assert expected_value(y_true, catch_none, **kwargs) == 0.0


def test_expected_value_penalises_false_positives():
    y_true = np.array([0, 0, 0, 0])
    flag_all = np.array([1, 1, 1, 1])
    value = expected_value(
        y_true, flag_all, offer_cost=50, churn_loss=500, offer_success_rate=0.3
    )
    assert value == -200.0


def test_tune_threshold_lowers_cutoff_when_misses_are_costly():
    """With a high churn_loss, the optimiser should cast a wider net than 0.5."""
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.25, 2000)
    # Probabilities correlated with the label but far from perfect.
    proba = np.clip(y * 0.35 + rng.normal(0.3, 0.2, 2000), 0.01, 0.99)

    result = tune_threshold(
        y, proba, offer_cost=10, churn_loss=1000, offer_success_rate=0.5
    )
    assert 0.0 < result["threshold"] < 0.5
    assert result["value_uplift_vs_0.5"] >= 0


def test_tune_threshold_is_never_worse_than_default():
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.3, 1000)
    proba = np.clip(y * 0.4 + rng.normal(0.3, 0.25, 1000), 0.01, 0.99)
    result = tune_threshold(y, proba)
    assert result["expected_value"] >= result["expected_value_at_0.5"]
    assert result["assumptions"]["offer_cost"] > 0


def test_feature_spec_roundtrips(cfg, sample_df):
    spec = build_spec(sample_df, cfg)
    restored = FeatureSpec.from_dict(spec.to_dict())
    assert restored.columns == spec.columns
    assert restored.categorical == spec.categorical


def test_feature_spec_example_is_complete(cfg, sample_df):
    spec = build_spec(sample_df, cfg)
    example = spec.example()
    assert set(example) == set(spec.columns)
    for col, values in spec.categorical.items():
        assert example[col] in values


def test_feature_spec_captures_observed_categories(cfg, sample_df):
    spec = build_spec(sample_df, cfg)
    assert set(spec.categorical["Contract"]) == set(sample_df["Contract"].unique())


def test_build_spec_ignores_unlisted_columns(cfg, sample_df):
    df = pd.DataFrame(sample_df)
    df["SomeUntrackedColumn"] = "x"
    spec = build_spec(df, cfg)
    assert "SomeUntrackedColumn" not in spec.columns
