"""Evaluation metrics and cost-sensitive threshold selection.

Churn is class-imbalanced and asymmetric in cost: missing a customer who leaves
is far more expensive than sending a retention offer to someone who would have
stayed. So we report a spread of metrics rather than accuracy, and we choose the
decision threshold by expected business value instead of defaulting to 0.5.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_rate": float(y_pred.mean()),
    }


def expected_value(
    y_true,
    y_pred,
    *,
    offer_cost: float,
    churn_loss: float,
    offer_success_rate: float,
) -> float:
    """Net value of running a retention campaign on the flagged customers.

    Measured against the do-nothing baseline:

    * flagging a true churner  -> we pay ``offer_cost`` and retain them with
      probability ``offer_success_rate``, saving ``churn_loss``.
    * flagging a non-churner   -> we pay ``offer_cost`` for nothing.
    * missing a churner        -> no cost change vs. doing nothing.

    So: ``value = TP * (success_rate * churn_loss - offer_cost) - FP * offer_cost``
    """
    _tn, fp, _fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tp * (offer_success_rate * churn_loss - offer_cost) - fp * offer_cost)


def tune_threshold(
    y_true,
    y_proba,
    *,
    offer_cost: float = 50.0,
    churn_loss: float = 500.0,
    offer_success_rate: float = 0.30,
    n_steps: int = 99,
) -> dict:
    """Pick the threshold maximising expected campaign value.

    Returns the chosen threshold plus the value curve endpoints, so the choice
    is auditable rather than a magic number.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    grid = np.linspace(0.01, 0.99, n_steps)
    values = np.array(
        [
            expected_value(
                y_true,
                (y_proba >= t).astype(int),
                offer_cost=offer_cost,
                churn_loss=churn_loss,
                offer_success_rate=offer_success_rate,
            )
            for t in grid
        ]
    )

    best_idx = int(np.argmax(values))
    best_t = float(grid[best_idx])
    default_value = expected_value(
        y_true,
        (y_proba >= 0.5).astype(int),
        offer_cost=offer_cost,
        churn_loss=churn_loss,
        offer_success_rate=offer_success_rate,
    )

    return {
        "threshold": round(best_t, 3),
        "expected_value": float(values[best_idx]),
        "expected_value_at_0.5": float(default_value),
        "value_uplift_vs_0.5": float(values[best_idx] - default_value),
        "assumptions": {
            "offer_cost": offer_cost,
            "churn_loss": churn_loss,
            "offer_success_rate": offer_success_rate,
        },
    }
