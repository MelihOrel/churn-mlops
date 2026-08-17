"""Unified command-line entry point tying the pipeline stages together.

    churn generate   # create synthetic data
    churn train      # train + log to MLflow + save artifact
    churn evaluate   # print metrics for the saved model on a fresh split
    churn monitor    # run drift report (reference vs current)
    churn predict    # score a JSON record from stdin or --json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

from .config import load_config


def _cmd_generate(args) -> int:
    """Prepare the active dataset: real extract for telco, generator otherwise."""
    cfg = load_config()
    script = "scripts/prepare_telco.py" if cfg.source == "telco" else "scripts/generate_data.py"
    return subprocess.call([sys.executable, script])


def _cmd_train(args) -> int:
    from .train import train

    train(load_config())
    return 0


def _cmd_evaluate(args) -> int:
    from .data import load_raw, split_features_target, validate
    from .evaluate import compute_metrics
    from .predict import predict_proba

    cfg = load_config()
    df = validate(load_raw(cfg.get("data.raw_path")), cfg)
    X, y = split_features_target(df, cfg)
    probs = predict_proba(X.to_dict(orient="records"), cfg)
    print(json.dumps(compute_metrics(y, probs), indent=2))
    return 0


def _cmd_monitor(args) -> int:
    from .monitoring import run_drift_report

    report = run_drift_report(load_config())
    return 1 if report["dataset_drift"] and args.fail_on_drift else 0


def _cmd_predict(args) -> int:
    from .predict import predict

    payload = args.json or sys.stdin.read()
    records = json.loads(payload)
    if isinstance(records, dict):
        records = [records]
    print(json.dumps(predict(records), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="churn", description="churn-mlops pipeline CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("generate", help="generate synthetic data").set_defaults(func=_cmd_generate)
    sub.add_parser("train", help="train the model").set_defaults(func=_cmd_train)
    sub.add_parser("evaluate", help="evaluate the saved model").set_defaults(func=_cmd_evaluate)

    p_mon = sub.add_parser("monitor", help="run drift report")
    p_mon.add_argument("--fail-on-drift", action="store_true", help="exit 1 if drift found")
    p_mon.set_defaults(func=_cmd_monitor)

    p_pred = sub.add_parser("predict", help="score record(s) from --json or stdin")
    p_pred.add_argument("--json", type=str, default=None)
    p_pred.set_defaults(func=_cmd_predict)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
