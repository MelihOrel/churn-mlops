"""Feature engineering as a reusable scikit-learn ``ColumnTransformer``.

Keeping preprocessing inside the fitted pipeline (rather than mutating the
dataframe in ad-hoc steps) is the single most important MLOps habit here:
the exact same transformations that ran at training time are serialized with
the model and replayed at serving time, so there is no train/serve skew.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import Config


def build_preprocessor(cfg: Config) -> ColumnTransformer:
    numeric = cfg.get("features.numeric", [])
    categorical = cfg.get("features.categorical", [])

    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )
