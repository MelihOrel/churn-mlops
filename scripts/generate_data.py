"""Generate a synthetic — but realistic — telco churn dataset.

Using synthetic data keeps the whole project reproducible with zero external
downloads or credentials: anyone can clone the repo and run the full pipeline.
The generator bakes in genuine signal (month-to-month contracts, high monthly
charges, low tenure and many support tickets all raise churn probability) so
the trained model has something real to learn.

It also emits two extra snapshots used by the monitoring stage:
  * a ``reference`` slice (the distribution the model was trained on), and
  * a ``current`` slice with deliberately shifted behaviour, so drift
    detection has something to catch.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CONTRACTS = ["Month-to-month", "One year", "Two year"]
INTERNET = ["DSL", "Fiber optic", "No"]
PAYMENT = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_dataset(n: int, seed: int, drift: float = 0.0) -> pd.DataFrame:
    """Build ``n`` rows of telco customers with a churn label.

    ``drift`` (0..1) nudges the covariate distribution — higher values mean
    shorter tenure, pricier plans and more month-to-month contracts, which is
    what we feed the monitoring demo as the "current" batch.
    """
    rng = np.random.default_rng(seed)

    tenure = np.clip(
        rng.gamma(shape=2.0 - drift, scale=12.0, size=n), 0, 72
    ).round()
    monthly = np.clip(
        rng.normal(70 + 15 * drift, 30, size=n), 18, 130
    ).round(2)
    # Total charges roughly tracks tenure * monthly with noise.
    total = np.clip(tenure * monthly * rng.uniform(0.8, 1.1, size=n), 0, None).round(2)
    tickets = rng.poisson(1.5 + 1.5 * drift, size=n)

    p_m2m = 0.5 + 0.3 * drift
    contract = rng.choice(CONTRACTS, size=n, p=[p_m2m, (1 - p_m2m) * 0.55, (1 - p_m2m) * 0.45])
    internet = rng.choice(INTERNET, size=n, p=[0.35, 0.45, 0.20])
    payment = rng.choice(PAYMENT, size=n, p=[0.35, 0.20, 0.22, 0.23])
    premium = rng.choice(["Yes", "No"], size=n, p=[0.35, 0.65])

    # Latent churn score: the "physics" of the dataset.
    score = (
        -1.2
        + 1.8 * (contract == "Month-to-month")
        - 0.03 * tenure
        + 0.012 * monthly
        + 0.25 * tickets
        + 0.6 * (internet == "Fiber optic")
        + 0.4 * (payment == "Electronic check")
        - 0.5 * (premium == "Yes")
    )
    churn_prob = _sigmoid(score)
    churn = rng.binomial(1, churn_prob)

    return pd.DataFrame(
        {
            "customerID": [f"C{seed:02d}{i:06d}" for i in range(n)],
            "tenure": tenure.astype(int),
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "NumSupportTickets": tickets.astype(int),
            "Contract": contract,
            "InternetService": internet,
            "PaymentMethod": payment,
            "HasPremiumSupport": premium,
            "Churn": np.where(churn == 1, "Yes", "No"),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic telco churn data.")
    parser.add_argument("--n", type=int, default=8000, help="rows in the main dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/churn_raw.csv")
    parser.add_argument("--reference-out", type=str, default="data/churn_reference.csv")
    parser.add_argument("--current-out", type=str, default="data/churn_current.csv")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    main_df = make_dataset(args.n, args.seed)
    main_df.to_csv(args.out, index=False)
    print(f"Wrote {len(main_df):,} rows -> {args.out}")

    # Reference = same distribution as training; current = drifted batch.
    make_dataset(2000, args.seed + 1, drift=0.0).to_csv(args.reference_out, index=False)
    make_dataset(2000, args.seed + 2, drift=0.9).to_csv(args.current_out, index=False)
    print(f"Wrote drift snapshots -> {args.reference_out}, {args.current_out}")


if __name__ == "__main__":
    main()
