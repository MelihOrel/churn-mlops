"""FastAPI model-serving app.

The request schema is **generated from the feature contract stored with the
trained model**, not hand-written. That means the API's validation, its
OpenAPI docs and the model always agree by construction: retrain on a different
dataset and the endpoint's accepted fields change with it, with no code edits
and no chance of a stale hand-maintained schema.

Run locally with:

    uvicorn churn.api.main:app --reload
"""
from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model

from .. import __version__
from ..config import load_config
from ..predict import ModelNotFoundError, load_model, load_spec, predict

app = FastAPI(
    title="Churn Prediction API",
    version=__version__,
    description=(
        "Serve the churn model trained by the churn-mlops pipeline. "
        "The request schema below is generated from the trained model's feature contract."
    ),
)


def _build_request_model() -> type[BaseModel]:
    """Create the pydantic request model from the model's feature spec.

    Falls back to a permissive model when no trained artifact is present yet,
    so the app still boots (and ``/health`` still answers) on a cold container.
    """
    try:
        spec = load_spec()
    except (ModelNotFoundError, KeyError):
        return create_model(
            "Customer",
            __doc__="Permissive schema — no trained model available to derive a contract.",
            __base__=_PermissiveBase,
        )

    fields: dict[str, Any] = {}
    for col in spec.numeric:
        lo, hi = spec.numeric_ranges.get(col, [0.0, None])
        fields[col] = (
            float,
            Field(..., ge=max(0.0, lo - abs(lo) * 0.5), examples=[round((lo + (hi or lo)) / 2, 2)]),
        )
    for col, values in spec.categorical.items():
        if values:
            fields[col] = (Literal[tuple(values)], Field(..., examples=[values[0]]))
        else:
            fields[col] = (str, Field(...))

    return create_model("Customer", **fields)


class _PermissiveBase(BaseModel):
    """Accepts arbitrary fields; used only before a model has been trained."""

    model_config = {"extra": "allow"}


Customer = _build_request_model()


class PredictionResponse(BaseModel):
    churn_probability: float = Field(..., description="P(churn) in [0, 1]")
    churn_prediction: bool = Field(..., description="Probability >= decision threshold")
    risk_band: Literal["low", "medium", "high"]


class BatchRequest(BaseModel):
    customers: list[Customer]  # type: ignore[valid-type]


@app.get("/health")
def health() -> dict:
    """Readiness probe: reports whether a model artifact is loadable."""
    try:
        load_model()
        model_ready = True
    except ModelNotFoundError:
        model_ready = False
    return {"status": "ok", "version": __version__, "model_ready": model_ready}


@app.get("/metadata")
def metadata() -> dict:
    """Describe the model currently being served."""
    cfg = load_config()
    payload: dict[str, Any] = {
        "model_type": cfg.get("model.type"),
        "decision_threshold": cfg.get("serving.decision_threshold", 0.5),
    }
    try:
        spec = load_spec()
        payload.update(
            {
                "data_source": spec.source,
                "features": spec.columns,
                "n_features": len(spec.columns),
                "categorical_values": spec.categorical,
            }
        )
    except (ModelNotFoundError, KeyError):
        payload["features"] = cfg.feature_columns
        payload["model_ready"] = False
    return payload


@app.post("/predict", response_model=PredictionResponse)
def predict_one(customer: Customer) -> dict:  # type: ignore[valid-type]
    try:
        return predict([customer.model_dump()])[0]
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(req: BatchRequest) -> list[dict]:
    try:
        return predict([c.model_dump() for c in req.customers])
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
