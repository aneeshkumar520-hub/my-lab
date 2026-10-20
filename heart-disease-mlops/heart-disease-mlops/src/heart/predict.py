"""Inference helpers shared by the API, notebooks and tests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd

from . import config


def load_model(path: Path | str = config.MODEL_PATH):
    return joblib.load(path)


def load_metadata(path: Path | str = config.METADATA_PATH) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def predict_records(model, records: Iterable[dict]) -> list[dict]:
    """Score patient records. ``confidence`` is the probability of the predicted class."""
    frame = pd.DataFrame(list(records)).reindex(columns=config.FEATURES).astype(float)
    proba = model.predict_proba(frame)[:, 1]
    results = []
    for p in proba:
        label = int(p >= 0.5)
        results.append({
            "prediction": label,
            "label": "heart disease" if label else "no heart disease",
            "probability_disease": round(float(p), 4),
            "confidence": round(float(p if label else 1 - p), 4),
        })
    return results
