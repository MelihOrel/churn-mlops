# churn-mlops

**An end-to-end MLOps pipeline for customer-churn prediction, built on the real IBM Telco Customer Churn dataset — data validation, leakage control, experiment tracking, cost-sensitive thresholding, a self-describing serving API, and drift monitoring.**

![CI](https://github.com/USERNAME/churn-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/USERNAME/churn-mlops/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Code style: ruff](https://img.shields.io/badge/lint-ruff-orange)
![Tests](https://img.shields.io/badge/tests-40%20passing-brightgreen)

This repository is a compact but production-shaped reference for how a machine-learning model actually gets *operated*, not just trained. The modelling task — predicting which telecom customers will churn — is a familiar one; the point of the project is everything around the model: reproducibility, validation, leakage control, tracking, serving, and monitoring.

---

## Results on real data

Trained on the **IBM Telco Customer Churn** dataset — 7,043 customers, 19 features, 26.5% churn base rate.

| Metric | Value |
|---|---|
| ROC-AUC | **0.853** |
| PR-AUC | 0.672 |
| Recall | 0.765 |
| Precision | 0.545 |
| F1 | 0.636 |
| Decision threshold | 0.30 *(tuned, not default)* |

ROC-AUC of ~0.85 is where a well-built model on this dataset should land. Anything dramatically higher is a red flag — and this repo demonstrates exactly why, below.

---

## Two things this project does that most churn demos don't

### 1. It refuses the free lunch: target leakage

The IBM extract ships with columns derived from the outcome being predicted — `Churn Score` (a vendor propensity score computed *with* knowledge of who churned), `Churn Reason` (populated only for customers who already left), and `CLTV`. Dropping them into the feature set produces a spectacular offline model and a worthless production one, because for a live customer those values do not exist yet.

The adapter excludes them, every exclusion carries a written justification, and a test asserts they can never reach the model. `make leakage` proves the point empirically:

| Model | ROC-AUC | PR-AUC | Recall |
|---|---|---|---|
| **Honest feature set** | **0.8534** | 0.6724 | 0.5374 |
| With leaky columns added | 1.0000 | 1.0000 | 1.0000 |
| `Churn Score` alone | 0.9402 | 0.8669 | 0.5802 |

A perfect 1.00 across every metric is not a great model — it is a bug. That gap, `+0.147` ROC-AUC of pure fiction, is the single most expensive mistake in applied churn modelling, and catching it is worth more than any amount of hyperparameter tuning.

### 2. It picks the operating point by business value, not by 0.5

Churn is asymmetric: missing a customer who leaves costs far more than sending a retention offer to someone who would have stayed. So the threshold is chosen by maximising expected campaign value on held-out data, given assumptions that live in config where a stakeholder can change them:

```
value = TP × (offer_success_rate × churn_loss − offer_cost) − FP × offer_cost
```

| Threshold | Recall | Precision | Expected campaign value |
|---|---|---|---|
| 0.50 (default) | 0.537 | 0.661 | $14,950 |
| **0.30 (tuned)** | **0.765** | 0.545 | **$16,650** |

Trading some precision for recall catches 42% more churners and is worth **+$1,700** per 1,409-customer test batch under the stated assumptions. The chosen threshold is serialized *with the model*, so serving cannot silently operate at a different point than the one that was evaluated.

---

## Architecture

```
                        config/config.yaml  ── single source of truth
                                 │
   ┌──────────────┐   ┌──────────▼───────────┐   ┌──────────────┐   ┌──────────────────┐
   │  IBM Telco   │──▶│  adapter: canonical  │──▶│  validate    │──▶│    train         │
   │  .xlsx       │   │  schema, drop leaky  │   │  schema+range│   │  + threshold tune│
   └──────────────┘   └──────────────────────┘   └──────────────┘   └────────┬─────────┘
   ┌──────────────┐              ▲                                            │
   │  synthetic   │──────────────┘                          ┌─────────────────┼──────────┐
   │  generator   │                                         │                 │          │
   └──────────────┘                                  ┌──────▼──────┐   ┌──────▼───────┐  │
                                                     │   MLflow    │   │ model bundle │  │
                                                     │  tracking   │   │ + FeatureSpec│  │
                                                     └─────────────┘   └──────┬───────┘  │
                                                                              │          │
   ┌──────────────────┐    ┌───────────────────────┐               ┌──────────▼───────┐  │
   │ reference vs     │──▶ │ drift monitor (PSI)   │── retrain ───▶│  FastAPI /predict│◀─┘
   │ current batch    │    │ reports/drift_report  │   trigger     │  schema from spec│
   └──────────────────┘    └───────────────────────┘               └──────────────────┘
```

Two design decisions carry most of the weight:

**Preprocessing lives inside the fitted pipeline.** The serialized artifact is one sklearn `Pipeline` = imputation + scaling + one-hot + classifier. The transformations at serving time are provably the ones from training, which eliminates train/serve skew.

**The feature contract travels with the model.** Training captures the exact feature names, ranges and observed category values into a `FeatureSpec`, serialized alongside the pipeline. The API builds its pydantic request model *from that spec at startup* — so validation, OpenAPI docs and the model can never fall out of sync, and swapping datasets requires no API code changes.

---

## What it demonstrates

| MLOps concern | How it's covered |
|---|---|
| Reproducibility | Config-driven runs, fixed seeds, pinned deps, schema-validated input |
| Source abstraction | Adapter layer; the same pipeline runs on real or generated data |
| **Leakage control** | Outcome-derived columns excluded, justified, and test-enforced |
| Data validation | Schema / type / range / target gates that fail loudly before training |
| Domain-aware cleaning | The 11 blank `TotalCharges` rows are tenure-0 customers → `0.0`, not dropped |
| Experiment tracking | Params, metrics and model logged to **MLflow** |
| **Cost-sensitive decisions** | Threshold tuned on expected business value, shipped with the model |
| Serving | **FastAPI** with a schema generated from the model's own contract |
| Monitoring | **PSI** drift detection, verified in both directions |
| Quality gates | 40 `pytest` tests, `ruff`, **GitHub Actions** matrix CI |
| Containerization | Multi-stage `Dockerfile` + `docker-compose` (API + MLflow UI) |

---

## Quickstart

**Zero setup** — the pipeline runs immediately on generated data, no downloads:

```bash
pip install -e ".[dev]"
make synthetic    # generate → train → monitor
make test         # 40 tests
```

**With the real data** — reproduces the headline numbers above:

```bash
# 1. Download the dataset (see "Getting the data" below) to
#    data/raw/Telco_customer_churn.xlsx
make pipeline     # prepare → train → evaluate → monitor
make leakage      # reproduce the leakage experiment
make serve        # → http://localhost:8000/docs
```

`config.yaml`'s `data.source` selects the active dataset (`telco` by default, `synthetic` as the fallback); everything downstream is source-agnostic, so switching requires no code changes.

### Getting the data

The dataset is **not committed to this repo**. IBM publishes it as a sample dataset, but with no clear redistribution licence — [the question has been raised on IBM's own community forum](https://community.ibm.com/community/user/datascience/discussion/copyright-and-license-of-the-telco-customer-churn-dataset) without a definitive answer — so redistributing it here would be presumptuous. Downloading it yourself takes a minute:

1. Get IBM's **Telco customer churn** extract, available through [IBM's Community Accelerator catalog](https://community.ibm.com/accelerators/catalog/content/Customer-churn) and mirrored on Kaggle.
2. Save it as `data/raw/Telco_customer_churn.xlsx`.
3. Run `make pipeline`.

> **Which version?** This project needs the **33-column** extract — the one containing `Churn Score`, `CLTV` and `Latitude`/`Longitude`. The widely-mirrored 21-column CSV is a different, trimmed release and will not work, because the leakage demonstration depends on precisely the columns that release omits. You do not have to check by hand: the adapter validates the schema on load and fails with a clear message if the file is the wrong one.

If you would rather not download anything, `make synthetic` exercises every stage of the pipeline on generated data instead.

---

## Using the API

```bash
make serve
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "tenure": 2, "MonthlyCharges": 89.5, "TotalCharges": 179.0,
        "Gender": "Female", "SeniorCitizen": "No", "Partner": "No", "Dependents": "No",
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check"
      }'
```

```json
{ "churn_probability": 0.7968, "churn_prediction": true, "risk_band": "high" }
```

Endpoints: `GET /health` (readiness probe), `GET /metadata` (the served model's contract), `POST /predict`, `POST /predict/batch`. Because the schema is generated from the model, `/docs` always shows the correct fields and the valid values for every categorical — and unknown categories are rejected with a 422.

---

## Drift monitoring

Models decay when live data drifts from the training distribution. The monitor computes the **Population Stability Index** per feature between a reference snapshot and an incoming batch. It is verified in *both* directions, which matters — a detector that only ever fires is as useless as one that never does:

| Scenario | Result |
|---|---|
| i.i.d. holdout split | `stable` — 0 features flagged, max PSI **0.006** |
| Skewed acquisition cohort (`make drift-demo`) | `DRIFT DETECTED` — 13 features, led by `Contract` (1.59) and `tenure` (1.41) |

The skewed batch simulates a realistic business change: a marketing push acquiring shorter-tenure, month-to-month, fiber-optic customers. Wired into a scheduled job, `churn monitor --fail-on-drift` exits non-zero and becomes an automated retrain trigger.

---

## Project structure

```
churn-mlops/
├── config/config.yaml           # single source of truth; selects the dataset
├── data/raw/                    # where you place the extract (not tracked)
├── scripts/
│   ├── prepare_telco.py         # real extract → canonical CSV + drift snapshots
│   ├── generate_data.py         # synthetic fallback dataset
│   └── leakage_experiment.py    # proves why the leaky columns are excluded
├── src/churn/
│   ├── config.py                # typed config + multi-dataset resolution
│   ├── datasets.py              # IBM Telco adapter, leakage exclusions
│   ├── data.py                  # loading + validation gates
│   ├── features.py              # ColumnTransformer preprocessing
│   ├── schema.py                # FeatureSpec — the contract shipped with the model
│   ├── train.py                 # training, MLflow logging, threshold tuning
│   ├── evaluate.py              # metrics + cost-sensitive threshold selection
│   ├── predict.py               # model loading + inference
│   ├── monitoring.py            # PSI drift detection
│   ├── cli.py                   # `churn` command-line entry point
│   └── api/main.py              # FastAPI app with a generated request schema
├── tests/                       # 40 tests: data, leakage, schema, threshold, drift, API
├── .github/workflows/ci.yml     # lint + test matrix + both pipeline smoke tests
├── Dockerfile · docker-compose.yml · Makefile
└── docs/architecture.md
```

---

## From local pipeline to Google Cloud

Each component maps onto a managed GCP stack:

| This repo | Google Cloud |
|---|---|
| Canonical CSV + adapter | **BigQuery** tables / scheduled queries |
| `ColumnTransformer` | BigQuery SQL transforms or **Dataflow** |
| MLflow tracking | **Vertex AI** Experiments & Model Registry |
| `train.py` | **Vertex AI** Training / Pipelines (KFP) |
| FastAPI serving | **Cloud Run** or Vertex AI Endpoints (**GKE**) |
| PSI drift monitor | Vertex AI **Model Monitoring** + **Looker** dashboard |
| GitHub Actions | **Cloud Build** triggers |

Point `data.raw_path` at a BigQuery export and `mlflow.tracking_uri` at a remote server, and the same code runs against the cloud.

---

## Testing

```bash
make test    # 40 tests
make lint
```

Coverage spans validation edge cases, the leakage exclusions, adapter refusal to impute suspicious blanks, the feature-contract round trip, expected-value arithmetic, drift detection in both directions, and the full HTTP surface including generated-schema enforcement. The API tests are dataset-agnostic — they build their fixture model from whatever the active config declares, which is what proves the dynamic schema actually works.

CI runs the suite on Python 3.10/3.11/3.12 and separately smoke-tests both the synthetic and the real-data pipelines end to end.

---

## Data

**IBM Telco Customer Churn** — 7,043 customers, published by IBM as a sample dataset for churn analytics. Not redistributed here; see [Getting the data](#getting-the-data). All results in this README were produced from that extract and are reproducible by following those steps.

## License

MIT — see [LICENSE](LICENSE). The licence covers the code in this repository; the IBM dataset is not included and is subject to whatever terms IBM applies to it.
