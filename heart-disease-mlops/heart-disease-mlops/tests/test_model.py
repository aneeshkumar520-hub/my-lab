import json

import joblib
import numpy as np

from heart import config, train as train_mod
from heart.predict import load_model, predict_records


def test_split_is_stratified_and_disjoint(clean_df):
    X_tr, X_te, y_tr, y_te = train_mod.split(clean_df)
    assert len(X_tr) + len(X_te) == len(clean_df)
    assert abs(y_tr.mean() - y_te.mean()) < 0.03
    assert set(X_tr.index).isdisjoint(X_te.index)


def test_training_returns_metrics_in_valid_range(tiny_model):
    best = tiny_model["out"]["best"]
    for metric in train_mod.SCORING:
        assert 0.0 <= best["test"][metric] <= 1.0
        mean, std = best["nested_cv"][metric]
        assert 0.0 <= mean <= 1.0 and std >= 0.0
    assert best["test"]["roc_auc"] > 0.75          # sanity floor: far above chance


def test_artifacts_are_saved_and_reload(tiny_model, clean_df):
    meta = json.loads(tiny_model["meta"].read_text())
    assert meta["features"] == config.FEATURES and "test_metrics" in meta
    model = joblib.load(tiny_model["path"])
    proba = model.predict_proba(clean_df[config.FEATURES].astype(float))[:, 1]
    assert proba.shape == (len(clean_df),) and ((proba >= 0) & (proba <= 1)).all()


def test_training_is_deterministic(clean_df):
    a = train_mod.train(clean_df, fast=True, use_mlflow=False, save=False, n_jobs=1)["best"]["test"]
    b = train_mod.train(clean_df, fast=True, use_mlflow=False, save=False, n_jobs=1)["best"]["test"]
    assert a == b


def test_predict_records_schema(tiny_model, sample):
    res = predict_records(load_model(tiny_model["path"]), [sample, dict(sample, age=40, ca=None)])
    assert len(res) == 2
    for r in res:
        assert r["prediction"] in (0, 1)
        assert 0.5 <= r["confidence"] <= 1.0
        assert 0.0 <= r["probability_disease"] <= 1.0


def test_model_handles_missing_optional_fields(tiny_model, sample):
    rec = dict(sample, ca=None, thal=None)
    out = predict_records(load_model(tiny_model["path"]), [rec])[0]
    assert np.isfinite(out["probability_disease"])


def test_high_risk_profile_scores_above_low_risk(tiny_model):
    model = load_model(tiny_model["path"])
    high = dict(age=64, sex=1, cp=0, trestbps=160, chol=300, fbs=0, restecg=1, thalach=105,
                exang=1, oldpeak=3.5, slope=1, ca=3, thal=3)
    low = dict(age=35, sex=0, cp=2, trestbps=115, chol=190, fbs=0, restecg=1, thalach=185,
               exang=0, oldpeak=0.0, slope=2, ca=0, thal=2)
    p_high, p_low = [r["probability_disease"] for r in predict_records(model, [high, low])]
    assert p_high > p_low


def test_select_model_prefers_simpler_model_within_one_std():
    def res(name, mean, std):
        return {"name": name, "nested_cv": {"roc_auc": (mean, std)}}
    pick = train_mod.select_model([res("random_forest", 0.910, 0.036),
                                   res("logistic_regression", 0.907, 0.039),
                                   res("gradient_boosting", 0.900, 0.030)])
    assert pick["name"] == "logistic_regression"
    # a clearly better complex model must still win
    pick = train_mod.select_model([res("random_forest", 0.95, 0.01), res("logistic_regression", 0.85, 0.02)])
    assert pick["name"] == "random_forest"
