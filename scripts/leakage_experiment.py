"""Demonstrate, empirically, why the excluded columns had to be excluded.

The IBM Telco extract ships with columns derived from the outcome we are trying
to predict — most notably ``Churn Score`` (a vendor propensity score computed
with knowledge of who churned) and ``Churn Reason`` (populated only for
customers who already left). Including them produces a model that looks
excellent offline and is worthless in production, because at scoring time for a
live customer those values do not exist.

This script trains the same pipeline twice — once on the honest feature set and
once with the leaky columns added — and prints the gap. Run it with:

    python scripts/leakage_experiment.py
"""
from __future__ import annotations

import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import load_config
from churn.evaluate import compute_metrics

LEAKY = ["Churn Score", "Churn Reason", "CLTV"]


def _pipeline(numeric: list[str], categorical: list[str], seed: int) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("i", SimpleImputer(strategy="median")),
                              ("s", StandardScaler())]), numeric),
            ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                              ("o", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )
    return Pipeline(
        [("preprocess", pre),
         ("model", GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                              learning_rate=0.05, random_state=seed))]
    )


def run() -> dict:
    cfg = load_config()
    seed = cfg.get("project.random_seed", 42)
    raw = pd.read_excel(cfg.get("data.source_file"))

    y = (raw["Churn Label"] == "Yes").astype(int)
    numeric = list(cfg.get("features.numeric", []))
    categorical = list(cfg.get("features.categorical", []))

    # Rebuild the honest feature frame from the raw extract.
    from churn.datasets import load_telco

    honest = load_telco(cfg.get("data.source_file"))[numeric + categorical]

    results: dict[str, dict] = {}

    # --- 1. Honest model -------------------------------------------------
    Xtr, Xte, ytr, yte = train_test_split(
        honest, y, test_size=0.2, random_state=seed, stratify=y
    )
    pipe = _pipeline(numeric, categorical, seed).fit(Xtr, ytr)
    results["honest"] = compute_metrics(yte, pipe.predict_proba(Xte)[:, 1])

    # --- 2. Same model, plus the leaky columns ---------------------------
    leaked = honest.copy()
    leaked["ChurnScore"] = raw["Churn Score"].values
    leaked["CLTV"] = raw["CLTV"].values
    leaked["ChurnReason"] = raw["Churn Reason"].fillna("__none__").values

    num_leak = numeric + ["ChurnScore", "CLTV"]
    cat_leak = categorical + ["ChurnReason"]

    Xtr, Xte, ytr, yte = train_test_split(
        leaked, y, test_size=0.2, random_state=seed, stratify=y
    )
    pipe_leak = _pipeline(num_leak, cat_leak, seed).fit(Xtr, ytr)
    results["leaky"] = compute_metrics(yte, pipe_leak.predict_proba(Xte)[:, 1])

    # --- 3. Churn Score alone --------------------------------------------
    score_only = pd.DataFrame({"ChurnScore": raw["Churn Score"].values})
    Xtr, Xte, ytr, yte = train_test_split(
        score_only, y, test_size=0.2, random_state=seed, stratify=y
    )
    pipe_score = _pipeline(["ChurnScore"], [], seed).fit(Xtr, ytr)
    results["churn_score_only"] = compute_metrics(yte, pipe_score.predict_proba(Xte)[:, 1])

    print("\n=== Leakage experiment (IBM Telco) ===")
    print(f"{'model':<28}{'ROC-AUC':>10}{'PR-AUC':>10}{'recall':>10}")
    for name, m in results.items():
        print(f"{name:<28}{m['roc_auc']:>10.4f}{m['pr_auc']:>10.4f}{m['recall']:>10.4f}")
    gap = results["leaky"]["roc_auc"] - results["honest"]["roc_auc"]
    print(f"\nLeakage inflates ROC-AUC by {gap:+.4f} — an entirely fictitious gain.")
    print("`Churn Reason` alone nearly separates the classes: it only exists post-churn.\n")

    return results


if __name__ == "__main__":
    out = run()
    print(json.dumps(out, indent=2))
