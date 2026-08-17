"""Integration tests for the FastAPI serving app.

These are deliberately **dataset-agnostic**: the fixture model is built from
whatever feature lists the active config declares, and the request payload is
generated from the feature contract the model ships with. That means this suite
passes against the synthetic dataset or the real Telco one without edits — and
it is precisely what proves the dynamic request schema works.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn import predict as predict_mod
from churn.config import load_config
from churn.features import build_preprocessor
from churn.schema import build_spec


def _fake_frame(cfg, n: int = 60) -> pd.DataFrame:
    """Build a small frame matching the active config's feature schema."""
    rng = np.random.default_rng(0)
    data: dict[str, object] = {}
    for col in cfg.get("features.numeric", []):
        data[col] = rng.uniform(0, 100, n).round(2)
    for col in cfg.get("features.categorical", []):
        data[col] = rng.choice([f"{col}_A", f"{col}_B"], n)
    return pd.DataFrame(data)


def _train_fixture_model() -> None:
    cfg = load_config()
    X = _fake_frame(cfg)
    # Deterministic but learnable label so the classifier sees both classes.
    y = (X[cfg.get("features.numeric")[0]] > 50).astype(int).tolist()

    spec = build_spec(X, cfg)
    pipe = Pipeline(
        [("preprocess", build_preprocessor(cfg)), ("model", LogisticRegression())]
    )
    pipe.fit(X[spec.columns], y)

    registry = Path(cfg.get("model.registry_dir", "models"))
    registry.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipe,
            "features": spec.columns,
            "spec": spec.to_dict(),
            "threshold": 0.5,
        },
        registry / cfg.get("model.model_filename", "churn_model.joblib"),
    )
    predict_mod.load_model.cache_clear()


# The app builds its request model at import time from the stored contract, so
# the fixture model has to exist first.
_train_fixture_model()
api_main = importlib.import_module("churn.api.main")
importlib.reload(api_main)
client = TestClient(api_main.app)

_CFG = load_config()
_SPEC = predict_mod.load_spec()
_PAYLOAD = _SPEC.example()


def test_health_reports_model_ready():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_ready"] is True


def test_metadata_reflects_trained_contract():
    body = client.get("/metadata").json()
    assert body["features"] == _SPEC.columns
    assert body["n_features"] == len(_SPEC.columns)
    assert body["data_source"] == _CFG.source


def test_predict_single_returns_probability():
    resp = client.post("/predict", json=_PAYLOAD)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_band"] in {"low", "medium", "high"}
    assert isinstance(body["churn_prediction"], bool)


def test_predict_rejects_missing_fields():
    assert client.post("/predict", json={}).status_code == 422


def test_predict_rejects_unknown_category():
    """The schema is derived from observed categories, so junk must be refused."""
    bad = dict(_PAYLOAD)
    cat_cols = list(_SPEC.categorical)
    if not cat_cols:
        return
    bad[cat_cols[0]] = "__definitely_not_a_real_category__"
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_rejects_negative_numeric():
    numeric = _SPEC.numeric
    if not numeric:
        return
    bad = dict(_PAYLOAD)
    bad[numeric[0]] = -9999.0
    assert client.post("/predict", json=bad).status_code == 422


def test_batch_prediction():
    resp = client.post("/predict/batch", json={"customers": [_PAYLOAD, _PAYLOAD]})
    assert resp.status_code == 200, resp.text
    assert len(resp.json()) == 2


def test_openapi_documents_generated_schema():
    """The generated contract must actually surface in the OpenAPI docs."""
    schema = client.get("/openapi.json").json()
    props = schema["components"]["schemas"]["Customer"]["properties"]
    for col in _SPEC.columns:
        assert col in props
