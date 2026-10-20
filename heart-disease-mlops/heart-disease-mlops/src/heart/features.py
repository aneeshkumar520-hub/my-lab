"""Feature engineering and the preprocessing transformer (single source of truth)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


class FeatureAdder(BaseEstimator, TransformerMixin):
    """Adds two clinically motivated features to a DataFrame.

    * ``pct_max_hr``: achieved max heart rate as a fraction of the age-predicted
      maximum (220 - age). Low chronotropic response is a risk marker.
    * ``st_depression_with_angina``: ST depression multiplied by the angina flag,
      capturing the interaction of two strong exercise-test signals.
    """

    def fit(self, X, y=None):  # noqa: D401 - sklearn API
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        age = X["age"].astype(float)
        X["pct_max_hr"] = X["thalach"].astype(float) / np.clip(220.0 - age, 1.0, None)
        X["st_depression_with_angina"] = X["oldpeak"].astype(float) * X["exang"].astype(float)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(config.FEATURES + config.ENGINEERED)


def build_preprocessor() -> Pipeline:
    """FeatureAdder -> (impute + scale numerics | impute + one-hot categoricals)."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    columns = ColumnTransformer([
        ("num", numeric, config.NUMERIC + config.ENGINEERED),
        ("cat", categorical, config.CATEGORICAL),
    ])
    return Pipeline([("features", FeatureAdder()), ("columns", columns)])


def feature_names(fitted_preprocessor: Pipeline) -> list[str]:
    """Readable names of the transformed columns of a fitted preprocessor."""
    ct = fitted_preprocessor.named_steps["columns"]
    return [n.replace("num__", "").replace("cat__", "") for n in ct.get_feature_names_out()]
