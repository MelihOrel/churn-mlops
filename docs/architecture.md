# Architecture & design notes

This document explains the *why* behind the structure. The guiding principle is
that an ML system's value in production comes less from the model and more from
the discipline around it: reproducibility, validation, leakage control, safe
serving, and monitoring.

## 1. Configuration as the single source of truth

Every stage reads from `config/config.yaml` — dataset selection, feature lists,
model hyperparameters, business cost assumptions, drift threshold. Nothing is
hard-coded in pipeline logic. A run is therefore fully described by
*(code version + config)*, which is the minimum bar for reproducibility.

`data.source` picks the active dataset and its block is merged up into `data`
and `features` at load time, so no downstream stage branches on source. Adding a
third dataset means adding a config block and an adapter — no changes to
training, serving or monitoring code.

## 2. Adapters isolate source-specific mess

`datasets.py` is the only place that knows a vendor extract exists. It maps raw
columns to the canonical schema, handles source-specific quirks, and drops
columns that must never be modelled. Everything downstream sees a clean frame.

Two quirks in the IBM Telco data are handled explicitly rather than swept up by
a generic cleaner:

- **Blank `Total Charges`.** Eleven rows hold `" "`. Every one has `tenure == 0`
  — customers who have not been billed yet, so the correct value is `0.0`. The
  adapter asserts that condition and *refuses to impute* if a blank ever shows
  up at non-zero tenure, because that would be a genuine data-quality incident,
  not a known quirk. A median impute here would invent revenue history; dropping
  the rows would bias against new customers, who are the highest-churn segment.
- **"No internet service" / "No phone service"** are preserved as their own
  categories rather than collapsed into `"No"`. "Declined the add-on" and
  "cannot have the add-on" are different customer states with different churn
  behaviour.

## 3. Leakage control as a first-class concern

The single highest-value decision in this project is what *not* to feed the
model. `TELCO_EXCLUDED` lists every dropped column with a written reason, split
into target leakage, identifiers, zero-variance constants, and high-cardinality
geography deferred to a future model.

`Churn Score` and `Churn Reason` are outright leakage — the first is a vendor
propensity score computed with outcome knowledge, the second exists only for
customers who already churned. `CLTV` is excluded as leakage-risk: a
vendor-derived lifetime value of ambiguous provenance. Including all three
yields a perfect 1.00 ROC-AUC, which `scripts/leakage_experiment.py` reproduces
on demand. Tests assert the exclusions hold, so a future refactor cannot quietly
reintroduce them.

## 4. Validation before training

`data.validate()` enforces schema, numeric types, non-negative ranges and a
clean Yes/No target before a model sees anything. Note that `drop_duplicates`
is **off** for the real dataset: two customers can legitimately share every
attribute once the identifier is removed, so dropping exact duplicates would
bias the base rate. It stays on for generated data, where duplicates are noise.
That distinction is a config value, not a hidden default.

## 5. Preprocessing inside the model object

The fitted estimator is one sklearn `Pipeline`: `ColumnTransformer` (impute +
scale + one-hot) followed by the classifier. Because preprocessing is *part of*
the serialized artifact, serving-time transformations are provably identical to
training-time ones. This eliminates train/serve skew — the classic source of
"great offline, broken in production". Unseen categories are absorbed via
`handle_unknown="ignore"`.

## 6. The feature contract travels with the model

`schema.py` captures feature names, numeric ranges and observed category values
into a `FeatureSpec` at training time, serialized alongside the pipeline. The
API builds its pydantic request model *from that spec* at startup rather than
from a hand-written class.

This is the difference between an API that documents the model and an API that
*is* the model's contract. Retrain on a different dataset and the endpoint's
accepted fields, valid categories and OpenAPI docs all change with it — no code
edits, no possibility of a stale schema silently accepting fields the model
never saw.

## 7. The operating point is a business decision

A 0.5 threshold is a default, not an answer. Churn is asymmetric: missing a
churner costs a customer's remaining lifetime value; a false positive costs one
retention offer. `evaluate.tune_threshold` maximises

```
value = TP × (offer_success_rate × churn_loss − offer_cost) − FP × offer_cost
```

over a threshold grid on held-out data. The assumptions are config values
because they are business inputs a stakeholder should be able to change without
touching code — and the returned object includes the value curve endpoints so
the choice is auditable rather than a magic number.

The chosen threshold is stored *in the model bundle*, so serving cannot operate
at a different point than the one that was evaluated. On the Telco data this
moves the threshold to 0.30, lifting recall from 0.537 to 0.765 for +$1,700 of
expected campaign value per test batch.

## 8. Monitoring, verified in both directions

`monitoring.py` computes the Population Stability Index per feature between a
reference distribution and a current batch. PSI is dependency-free,
interpretable, and works for numeric (quantile-binned) and categorical features
alike.

Critically, the monitor is validated for **both** outcomes: on an i.i.d. split
of the real data it reports stable with a max PSI of 0.006 (no false alarms),
and on a deliberately skewed acquisition cohort it flags 13 features led by
exactly the dimensions that were skewed. A detector that always fires is as
useless as one that never does.

`churn monitor --fail-on-drift` returns a non-zero exit code so it can gate a
scheduled job and trigger a retrain.

## Trade-offs & next steps

- **Geography is unused.** `City`, `Zip Code` and lat/long are excluded from the
  baseline; a regional model with target or geo encoding is the natural
  extension.
- **Batch drift only.** Prediction-distribution and concept drift (label delay
  aware) would be the next monitoring additions.
- **Registry stages.** Model promotion (staging → production) with automated
  retrain-on-drift belongs here once a scheduler (Cloud Composer / Vertex
  Pipelines) is in the picture.
- **Cost assumptions are illustrative.** The `business:` block uses plausible
  round numbers; in a real engagement these come from finance, and the threshold
  should be re-tuned whenever they change.
