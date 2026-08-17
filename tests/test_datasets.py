"""Tests for the IBM Telco adapter — especially the leakage guarantees.

The most valuable test in this file is the one asserting that the vendor's
outcome-derived columns never survive into the modelling frame. That is the
kind of mistake that produces a 1.00 AUC and a useless production model.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from churn.config import Config, ConfigError, _resolve_active_dataset, load_config
from churn.datasets import (
    TELCO_COLUMN_MAP,
    TELCO_EXCLUDED,
    DatasetAdapterError,
    leakage_report,
    load_telco,
)

RAW = Path("data/raw/Telco_customer_churn.xlsx")
requires_raw = pytest.mark.skipif(
    not RAW.exists(), reason="raw Telco extract not present in this checkout"
)

LEAKY_COLUMNS = ["Churn Score", "Churn Reason", "Churn Value", "CLTV"]


def test_leakage_columns_are_documented():
    """Every excluded column must carry a written justification."""
    report = leakage_report()
    for col in LEAKY_COLUMNS:
        assert col in report
        assert report[col].strip(), f"{col} has no documented reason"


def test_known_leaky_columns_are_flagged_as_leakage():
    report = leakage_report()
    for col in ("Churn Score", "Churn Reason"):
        assert "leakage" in report[col].lower()


@requires_raw
def test_adapter_drops_every_excluded_column():
    df = load_telco(RAW)
    for col in TELCO_EXCLUDED:
        assert col not in df.columns, f"excluded column '{col}' leaked into the frame"


@requires_raw
def test_adapter_removes_outcome_derived_columns():
    """Explicit guard against the highest-risk failure mode."""
    df = load_telco(RAW)
    lowered = {c.lower().replace(" ", "") for c in df.columns}
    for banned in ("churnscore", "churnreason", "churnvalue", "cltv"):
        assert banned not in lowered


@requires_raw
def test_adapter_renames_to_canonical_schema():
    df = load_telco(RAW)
    for canonical in TELCO_COLUMN_MAP.values():
        assert canonical in df.columns
    assert "Churn" in df.columns
    assert set(df["Churn"].unique()) == {"Yes", "No"}


@requires_raw
def test_total_charges_is_numeric_and_zero_filled():
    """The 11 blank TotalCharges rows are tenure-0 customers -> 0.0, not NaN."""
    df = load_telco(RAW)
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])
    assert df["TotalCharges"].isna().sum() == 0
    zero_rows = df[df["TotalCharges"] == 0.0]
    assert (zero_rows["tenure"] == 0).all()


@requires_raw
def test_row_count_and_base_rate_are_preserved():
    df = load_telco(RAW)
    assert len(df) == 7043
    assert 0.25 < (df["Churn"] == "Yes").mean() < 0.28


def test_adapter_rejects_a_file_that_is_not_telco(tmp_path):
    bogus = tmp_path / "not_telco.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(bogus, index=False)
    with pytest.raises(DatasetAdapterError, match="does not look like"):
        load_telco(bogus)


def test_adapter_refuses_to_impute_blanks_at_nonzero_tenure(tmp_path):
    """Zero-filling is only valid for never-billed customers; anything else fails."""
    bad = pd.DataFrame(
        {
            "Churn Label": ["No"],
            "Tenure Months": [24],          # billed for two years...
            "Monthly Charges": [70.0],
            "Total Charges": [" "],         # ...but no total: a real problem
        }
    )
    path = tmp_path / "bad.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(DatasetAdapterError, match="refusing to impute"):
        load_telco(path)


def test_missing_file_gives_actionable_error():
    """A missing extract must tell the user how to fix it, not just fail."""
    with pytest.raises(FileNotFoundError) as exc:
        load_telco("data/raw/does_not_exist.xlsx")
    message = str(exc.value)
    assert "does_not_exist.xlsx" in message      # says which file
    assert "Churn Score" in message              # says which version is needed
    assert "synthetic" in message                # offers the no-download path


# --- multi-dataset config resolution ---------------------------------------


def test_config_resolves_active_dataset():
    raw = {
        "data": {"source": "b", "target": "Churn"},
        "datasets": {
            "a": {"raw_path": "a.csv", "features": {"numeric": ["x"], "categorical": []}},
            "b": {"raw_path": "b.csv", "features": {"numeric": ["y"], "categorical": ["z"]}},
        },
    }
    cfg = Config(raw=_resolve_active_dataset(raw))
    assert cfg.get("data.raw_path") == "b.csv"
    assert cfg.feature_columns == ["y", "z"]
    assert cfg.source == "b"


def test_config_rejects_unknown_source():
    raw = {"data": {"source": "nope"}, "datasets": {"a": {"features": {"numeric": []}}}}
    with pytest.raises(ConfigError, match="no matching entry"):
        _resolve_active_dataset(raw)


def test_shipped_config_is_valid():
    cfg = load_config()
    assert cfg.source in {"telco", "synthetic"}
    assert len(cfg.feature_columns) > 0
