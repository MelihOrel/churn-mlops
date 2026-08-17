"""Training stage: fit the pipeline, log everything to MLflow, register artifact.

The full estimator is a single sklearn ``Pipeline`` = preprocessing + model, so
the serialized ``.joblib`` is completely self-contained. Alongside it we store
the ``FeatureSpec`` — the feature contract the serving layer validates against.
Metrics, params and the model are all logged to MLflow for run comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import Config, load_config
from .data import load_raw, split_features_target, validate
from .evaluate import compute_metrics, tune_threshold
from .features import build_preprocessor
from .schema import build_spec


def _build_estimator(cfg: Config):
    model_type = cfg.get("model.type", "gradient_boosting")
    seed = cfg.get("project.random_seed", 42)
    if model_type == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            random_state=seed,
            class_weight="balanced" if cfg.get("model.class_weight_balanced") else None,
        )
    return GradientBoostingClassifier(
        n_estimators=cfg.get("model.n_estimators", 200),
        max_depth=cfg.get("model.max_depth", 3),
        learning_rate=cfg.get("model.learning_rate", 0.05),
        random_state=seed,
    )


def train(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    seed = cfg.get("project.random_seed", 42)

    df = validate(load_raw(cfg.get("data.raw_path")), cfg)
    X, y = split_features_target(df, cfg)

    # Stratified split preserves the ~27% churn base rate in both halves.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.get("data.test_size", 0.2), random_state=seed, stratify=y
    )

    spec = build_spec(X_train, cfg)

    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(cfg)),
            ("model", _build_estimator(cfg)),
        ]
    )

    mlflow.set_tracking_uri(cfg.get("mlflow.tracking_uri", "sqlite:///mlflow.db"))
    mlflow.set_experiment(cfg.get("mlflow.experiment_name", "churn-prediction"))

    with mlflow.start_run() as run:
        pipeline.fit(X_train, y_train)

        proba = pipeline.predict_proba(X_test)[:, 1]

        # Choose the operating point by expected business value, not by 0.5.
        tuning = tune_threshold(
            y_test,
            proba,
            offer_cost=cfg.get("business.retention_offer_cost", 50.0),
            churn_loss=cfg.get("business.churn_loss", 500.0),
            offer_success_rate=cfg.get("business.offer_success_rate", 0.30),
        )
        configured = cfg.get("serving.decision_threshold", "auto")
        threshold = tuning["threshold"] if configured == "auto" else float(configured)

        metrics = compute_metrics(y_test, proba, threshold=threshold)
        metrics["decision_threshold"] = threshold
        metrics["expected_campaign_value"] = tuning["expected_value"]

        mlflow.log_params(
            {
                "data_source": cfg.source,
                "model_type": cfg.get("model.type"),
                "n_estimators": cfg.get("model.n_estimators"),
                "max_depth": cfg.get("model.max_depth"),
                "learning_rate": cfg.get("model.learning_rate"),
                "n_features": len(spec.columns),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "base_churn_rate": round(float(y.mean()), 4),
            }
        )
        mlflow.log_metrics(metrics)
        # cloudpickle keeps the full sklearn Pipeline (incl. numpy dtypes)
        # loadable without skops' trusted-type gate.
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        # Persist a plain artifact so serving needs no MLflow server at runtime.
        registry_dir = Path(cfg.get("model.registry_dir", "models"))
        registry_dir.mkdir(parents=True, exist_ok=True)
        model_path = registry_dir / cfg.get("model.model_filename", "churn_model.joblib")
        # The threshold ships with the model so serving cannot silently use a
        # different operating point than the one that was evaluated.
        joblib.dump(
            {
                "pipeline": pipeline,
                "features": spec.columns,
                "spec": spec.to_dict(),
                "threshold": threshold,
                "threshold_tuning": tuning,
            },
            model_path,
        )

        spec_path = registry_dir / cfg.get("model.spec_filename", "feature_spec.json")
        spec_path.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")

        metrics_path = registry_dir / cfg.get("model.metrics_filename", "metrics.json")
        metrics_out = {
            **metrics,
            "data_source": cfg.source,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "mlflow_run_id": run.info.run_id,
        }
        metrics_path.write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")

        print(f"[train] source={cfg.source} run_id={run.info.run_id}")
        print(f"[train] metrics={json.dumps(metrics, indent=2)}")
        print(
            f"[train] threshold={threshold} "
            f"(value uplift vs 0.5: {tuning['value_uplift_vs_0.5']:+,.0f})"
        )
        print(f"[train] model -> {model_path}")

    return metrics_out


if __name__ == "__main__":
    train()
