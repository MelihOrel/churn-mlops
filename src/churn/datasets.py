"""Source-specific dataset adapters.

Each adapter takes a raw third-party file and normalises it into the project's
canonical schema, so everything downstream (validation, features, training,
serving) is source-agnostic. Source-specific quirks are handled *here* and
nowhere else — that separation is what lets the same pipeline run against a
synthetic generator or a real vendor extract without branching logic.

Currently supported:
  * ``telco`` — the IBM Telco Customer Churn dataset (7,043 rows).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# IBM Telco Customer Churn
# ---------------------------------------------------------------------------

# Raw column -> canonical column. Renaming here keeps spaces and vendor-specific
# naming out of the model schema and the serving API.
TELCO_COLUMN_MAP: dict[str, str] = {
    "Tenure Months": "tenure",
    "Monthly Charges": "MonthlyCharges",
    "Total Charges": "TotalCharges",
    "Senior Citizen": "SeniorCitizen",
    "Phone Service": "PhoneService",
    "Multiple Lines": "MultipleLines",
    "Internet Service": "InternetService",
    "Online Security": "OnlineSecurity",
    "Online Backup": "OnlineBackup",
    "Device Protection": "DeviceProtection",
    "Tech Support": "TechSupport",
    "Streaming TV": "StreamingTV",
    "Streaming Movies": "StreamingMovies",
    "Paperless Billing": "PaperlessBilling",
    "Payment Method": "PaymentMethod",
    "Churn Label": "Churn",
}

# Columns that must never reach the model, with the reason. Documenting the
# *why* matters more than the list: two of these are genuine target leakage and
# would silently inflate every offline metric.
TELCO_EXCLUDED: dict[str, str] = {
    # --- Target leakage: derived from or after the outcome we predict ---
    "Churn Score": "leakage: vendor propensity score computed with outcome knowledge",
    "Churn Reason": "leakage: populated only for customers who already churned",
    "CLTV": "leakage-risk: vendor-derived lifetime value of ambiguous provenance",
    "Churn Value": "target duplicate (numeric encoding of Churn Label)",
    # --- Identifiers ---
    "CustomerID": "identifier: unique per row, no predictive content",
    # --- Zero-variance constants ---
    "Count": "constant column (always 1)",
    "Country": "constant column (United States)",
    "State": "constant column (California)",
    # --- High-cardinality geography: excluded from the baseline model ---
    "City": "high cardinality (1,129 values); needs target/geo encoding",
    "Zip Code": "high cardinality; would be treated as a meaningless number",
    "Lat Long": "redundant with Latitude/Longitude",
    "Latitude": "geo feature reserved for a future regional model",
    "Longitude": "geo feature reserved for a future regional model",
}


class DatasetAdapterError(ValueError):
    """Raised when a raw source file does not look like the expected dataset."""


def load_telco(path: str | Path) -> pd.DataFrame:
    """Load the IBM Telco churn file (.xlsx or .csv) into the canonical schema.

    Handles the two well-known quirks of this dataset explicitly:

    1. ``Total Charges`` is stored as text and contains 11 blank strings. Every
       one of those rows has ``tenure == 0`` — they are customers who have not
       been billed yet, so the correct value is ``0.0``, not a median impute
       and not a dropped row.
    2. Service columns encode "not subscribed" as ``"No internet service"`` /
       ``"No phone service"``. These are kept as their own category rather than
       collapsed into ``"No"``, because "declined the add-on" and "cannot have
       the add-on" are different customer states.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Telco source file not found at '{path}'.\n\n"
            "This repo does not ship the dataset — IBM publishes it as a sample "
            "without clear redistribution terms.\n"
            "  1. Download IBM's 'Telco customer churn' extract (the 33-column "
            "version containing 'Churn Score' and 'CLTV').\n"
            f"  2. Save it as '{path}'.\n"
            "  See the README section 'Getting the data' for links.\n\n"
            "Or skip it entirely and run on generated data:\n"
            "  make synthetic     (equivalently: CHURN_DATA_SOURCE=synthetic)"
        )

    raw = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)

    required = {"Churn Label", "Tenure Months", "Monthly Charges", "Total Charges"}
    missing = required - set(raw.columns)
    if missing:
        raise DatasetAdapterError(
            f"'{path.name}' does not look like the IBM Telco dataset; missing {sorted(missing)}"
        )

    df = raw.rename(columns=TELCO_COLUMN_MAP)

    # Quirk 1: blank Total Charges for never-billed customers.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    unbilled = df["TotalCharges"].isna()
    if unbilled.any():
        tenures = sorted(df.loc[unbilled, "tenure"].unique())
        if tenures != [0]:
            # Only tenure-0 rows are safe to zero-fill; anything else is a real
            # data-quality problem and should not be silently patched.
            raise DatasetAdapterError(
                f"Blank TotalCharges found at non-zero tenure {tenures}; refusing to impute."
            )
        df.loc[unbilled, "TotalCharges"] = 0.0
        print(
            f"[telco] filled {int(unbilled.sum())} blank TotalCharges with 0.0 "
            "(tenure-0 customers, not yet billed)"
        )

    # Drop identifiers, constants, geography and — critically — leakage columns.
    to_drop = [c for c in TELCO_EXCLUDED if c in df.columns]
    df = df.drop(columns=to_drop)
    print(f"[telco] dropped {len(to_drop)} non-modelling column(s); see TELCO_EXCLUDED for reasons")

    return df.reset_index(drop=True)


def leakage_report() -> dict[str, str]:
    """Return the excluded-column decisions, for docs and tests."""
    return dict(TELCO_EXCLUDED)
