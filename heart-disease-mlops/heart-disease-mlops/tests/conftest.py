"""Shared fixtures. Tests never touch the network or the production model."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heart import config, data  # noqa: E402

SAMPLE = {"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0,
          "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return data.load_raw(config.DATA_RAW)


@pytest.fixture(scope="session")
def clean_df(raw_df) -> pd.DataFrame:
    return data.clean(raw_df)


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory, clean_df):
    """Train a quick model into a temp dir so API tests are independent of models/."""
    from heart import train as train_mod
    d = tmp_path_factory.mktemp("model")
    out = train_mod.train(clean_df, fast=True, use_mlflow=False, save=True,
                          model_path=d / "m.joblib", metadata_path=d / "meta.json", n_jobs=1)
    return {"path": d / "m.joblib", "meta": d / "meta.json", "out": out}


@pytest.fixture()
def sample():
    return dict(SAMPLE)
