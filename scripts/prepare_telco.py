"""Turn the raw IBM Telco extract into the canonical dataset the pipeline uses.

Also writes the two snapshots the drift monitor compares:

* ``reference`` — the distribution the model is trained on.
* ``current``   — an incoming batch. By default this is an i.i.d. split of the
  same data, which *should* show no drift; that is the honest baseline and it
  proves the monitor does not fire on noise. Pass ``--drift-scenario`` to build
  a deliberately skewed batch (a month-to-month, short-tenure, fiber-heavy
  acquisition cohort) to see detection working on real data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from churn.config import load_config
from churn.datasets import load_telco


def build_drift_batch(df: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    """Sample a skewed cohort: newer, month-to-month, fiber-optic customers.

    This mimics a realistic business change — a marketing push that acquires a
    different kind of customer than the model was trained on.
    """
    rng = np.random.default_rng(seed)
    weights = np.ones(len(df), dtype=float)
    weights *= np.where(df["Contract"].eq("Month-to-month"), 4.0, 0.4)
    weights *= np.where(df["tenure"] <= 12, 3.5, 0.5)
    weights *= np.where(df["InternetService"].eq("Fiber optic"), 2.5, 0.6)
    weights /= weights.sum()
    idx = rng.choice(len(df), size=min(size, len(df)), replace=True, p=weights)
    return df.iloc[idx].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the IBM Telco churn dataset.")
    parser.add_argument("--drift-scenario", action="store_true",
                        help="build a skewed 'current' batch instead of an i.i.d. split")
    parser.add_argument("--reference-frac", type=float, default=0.6)
    args = parser.parse_args()

    cfg = load_config()
    if cfg.source != "telco":
        raise SystemExit(
            f"config data.source is '{cfg.source}'; set it to 'telco' before running this."
        )

    seed = cfg.get("project.random_seed", 42)
    src = cfg.get("data.source_file")
    out = Path(cfg.get("data.raw_path"))
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        df = load_telco(src)
    except FileNotFoundError as exc:
        # A missing download is an expected first-run state, not a crash:
        # show the instructions, not a traceback.
        raise SystemExit(f"\n{exc}\n") from None

    df.to_csv(out, index=False)
    churn_rate = (df["Churn"] == "Yes").mean()
    print(f"[prepare] {len(df):,} rows, churn rate {churn_rate:.1%} -> {out}")

    # Reference / current snapshots for the drift monitor.
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    cut = int(len(shuffled) * args.reference_frac)
    reference, holdout = shuffled.iloc[:cut], shuffled.iloc[cut:]

    reference.to_csv(cfg.get("data.reference_path"), index=False)

    if args.drift_scenario:
        current = build_drift_batch(df, size=len(holdout), seed=seed + 1)
        label = "skewed acquisition cohort (drift expected)"
    else:
        current = holdout.reset_index(drop=True)
        label = "i.i.d. holdout (no drift expected)"

    current.to_csv(cfg.get("data.current_path"), index=False)
    print(f"[prepare] reference={len(reference):,} rows, current={len(current):,} rows — {label}")


if __name__ == "__main__":
    main()
