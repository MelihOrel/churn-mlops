"""Shared pytest fixtures."""
from __future__ import annotations

import pandas as pd
import pytest

from churn.config import Config

_RAW_CONFIG = {
    "project": {"random_seed": 7},
    "data": {"target": "Churn", "test_size": 0.2},
    "features": {
        "numeric": ["tenure", "MonthlyCharges", "TotalCharges", "NumSupportTickets"],
        "categorical": [
            "Contract",
            "InternetService",
            "PaymentMethod",
            "HasPremiumSupport",
        ],
    },
    "model": {"type": "logistic_regression"},
}


@pytest.fixture
def cfg() -> Config:
    return Config(raw=_RAW_CONFIG)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tenure": [1, 40, 12, 60],
            "MonthlyCharges": [90.0, 55.0, 70.0, 40.0],
            "TotalCharges": [90.0, 2200.0, 840.0, 2400.0],
            "NumSupportTickets": [5, 0, 2, 1],
            "Contract": ["Month-to-month", "Two year", "One year", "Two year"],
            "InternetService": ["Fiber optic", "DSL", "Fiber optic", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Credit card",
                "Mailed check",
                "Bank transfer",
            ],
            "HasPremiumSupport": ["No", "Yes", "No", "Yes"],
            "Churn": ["Yes", "No", "Yes", "No"],
        }
    )
