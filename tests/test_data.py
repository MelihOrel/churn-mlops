"""Tests for data validation."""
import pandas as pd
import pytest

from churn.data import DataValidationError, split_features_target, validate


def test_validate_passes_on_clean_data(sample_df, cfg):
    out = validate(sample_df, cfg)
    assert len(out) == len(sample_df)


def test_validate_flags_missing_column(sample_df, cfg):
    broken = sample_df.drop(columns=["tenure"])
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate(broken, cfg)


def test_validate_flags_negative_values(sample_df, cfg):
    broken = sample_df.copy()
    broken.loc[0, "MonthlyCharges"] = -10.0
    with pytest.raises(DataValidationError, match="Negative values"):
        validate(broken, cfg)


def test_validate_flags_bad_target(sample_df, cfg):
    broken = sample_df.copy()
    broken.loc[0, "Churn"] = "Maybe"
    with pytest.raises(DataValidationError, match="must be Yes/No"):
        validate(broken, cfg)


def test_validate_coerces_numeric_strings(sample_df, cfg):
    df = sample_df.copy()
    df["TotalCharges"] = df["TotalCharges"].astype(str)
    out = validate(df, cfg)
    assert pd.api.types.is_numeric_dtype(out["TotalCharges"])


def test_split_features_target_encodes_label(sample_df, cfg):
    X, y = split_features_target(validate(sample_df, cfg), cfg)
    assert set(y.unique()) <= {0, 1}
    assert list(X.columns) == cfg.feature_columns
