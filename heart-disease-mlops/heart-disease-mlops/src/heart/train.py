"""Train, tune, evaluate and track the heart-disease classifiers.

Run:  python -m heart.train            (full grids, logs everything to MLflow)
      python -m heart.train --fast     (tiny grids; used by CI smoke checks/tests)

For every candidate model the script
  1. tunes hyper-parameters with GridSearchCV (stratified 5-fold, ROC-AUC),
  2. estimates generalisation with *nested* cross-validation on the training split
     (accuracy / precision / recall / F1 / ROC-AUC),
  3. evaluates the tuned model once on a held-out test split,
  4. logs params, metrics, plots and the model to MLflow (one child run each),
and finally saves the selected model (one-standard-error rule on nested-CV ROC-AUC)
as a reusable pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, classification_report,
    f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV, StratifiedKFold, cross_validate, train_test_split)
from sklearn.pipeline import Pipeline  # noqa: E402

from . import __version__, config, data  # noqa: E402
from .features import build_preprocessor, feature_names  # noqa: E402

log = logging.getLogger("heart.train")
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def candidates(fast: bool = False) -> dict:
    """Model name -> (estimator, hyper-parameter grid)."""
    rs = config.RANDOM_STATE
    if fast:
        return {
            "logistic_regression": (LogisticRegression(max_iter=2000, random_state=rs),
                                    {"model__C": [0.1, 1.0]}),
            "random_forest": (RandomForestClassifier(random_state=rs),
                              {"model__n_estimators": [50], "model__max_depth": [3, None]}),
        }
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=rs),
            {"model__C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
             "model__class_weight": [None, "balanced"]}),
        "random_forest": (
            RandomForestClassifier(random_state=rs),
            {"model__n_estimators": [100, 300],
             "model__max_depth": [3, 5, None],
             "model__min_samples_leaf": [1, 3, 5]}),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=rs),
            {"model__n_estimators": [50, 100, 200],
             "model__learning_rate": [0.03, 0.1],
             "model__max_depth": [2, 3]}),
    }


COMPLEXITY = {"logistic_regression": 0, "random_forest": 1, "gradient_boosting": 2}


def select_model(results: list[dict]) -> dict:
    """One-standard-error rule on nested-CV ROC-AUC (test set is never used).

    Take the best mean score, keep every model whose mean is within one standard
    deviation of it, and return the *simplest* of those (ties -> higher mean).
    On a 300-patient dataset differences smaller than the fold-to-fold spread are
    noise, so the simpler, more interpretable model is preferred.
    """
    top = max(results, key=lambda r: r["nested_cv"]["roc_auc"][0])
    mean, std = top["nested_cv"]["roc_auc"]
    eligible = [r for r in results if r["nested_cv"]["roc_auc"][0] >= mean - std]
    return min(eligible, key=lambda r: (COMPLEXITY.get(r["name"], 99), -r["nested_cv"]["roc_auc"][0]))


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])


def split(df: pd.DataFrame):
    X = df[config.FEATURES].astype(float)
    y = df[config.TARGET]
    return train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y,
                            random_state=config.RANDOM_STATE)


def test_metrics(pipe, X, y) -> dict:
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred),
            "roc_auc": roc_auc_score(y, proba)}


def evaluate_candidate(name, estimator, grid, X_tr, y_tr, X_te, y_te, n_jobs=-1) -> dict:
    """Tune + nested-CV + hold-out evaluation of one model."""
    pipe = make_pipeline(estimator)
    inner = StratifiedKFold(3, shuffle=True, random_state=config.RANDOM_STATE)
    outer = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)

    # (a) nested CV -> approximately unbiased estimate that includes the tuning step
    search_nested = GridSearchCV(pipe, grid, cv=inner, scoring="roc_auc", n_jobs=n_jobs)
    nested = cross_validate(search_nested, X_tr, y_tr, cv=outer, scoring=SCORING, n_jobs=1)
    cv = {m: (float(np.mean(nested[f"test_{m}"])), float(np.std(nested[f"test_{m}"])))
          for m in SCORING}

    # (b) final tuning on the whole training split (5-fold) + one hold-out evaluation
    search = GridSearchCV(pipe, grid, cv=outer, scoring="roc_auc", n_jobs=n_jobs, refit=True)
    search.fit(X_tr, y_tr)
    best = search.best_estimator_
    return {"name": name, "pipeline": best, "best_params": search.best_params_,
            "tuning_cv_roc_auc": float(search.best_score_), "nested_cv": cv,
            "test": test_metrics(best, X_te, y_te),
            "cv_results": pd.DataFrame(search.cv_results_)}


def make_plots(res: dict, X_te, y_te, outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    pipe, paths = res["pipeline"], []

    fig, ax = plt.subplots(figsize=(5, 4.5))
    RocCurveDisplay.from_estimator(pipe, X_te, y_te, ax=ax, name=res["name"])
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax.set_title(f"ROC curve (test) - {res['name']}")
    paths.append(outdir / "roc_curve.png")
    fig.savefig(paths[-1], dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ConfusionMatrixDisplay.from_estimator(pipe, X_te, y_te, ax=ax, cmap="Blues",
                                          display_labels=["No disease", "Disease"])
    ax.set_title(f"Confusion matrix (test) - {res['name']}")
    paths.append(outdir / "confusion_matrix.png")
    fig.savefig(paths[-1], dpi=140, bbox_inches="tight")
    plt.close(fig)

    model = pipe.named_steps["model"]
    names = feature_names(pipe.named_steps["preprocess"])
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        title = "Feature importance"
    else:
        imp = np.abs(model.coef_.ravel())
        title = "|Coefficient| (standardised features)"
    order = np.argsort(imp)[-15:]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(np.array(names)[order], imp[order], color="#4C78A8")
    ax.set_title(f"{title} - {res['name']}")
    paths.append(outdir / "feature_importance.png")
    fig.savefig(paths[-1], dpi=140, bbox_inches="tight")
    plt.close(fig)
    return paths


def comparison_plot(results: list[dict], path: Path) -> Path:
    labels = [r["name"] for r in results]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    w = 0.38
    ax.bar(x - w / 2, [r["nested_cv"]["roc_auc"][0] for r in results], w,
           yerr=[r["nested_cv"]["roc_auc"][1] for r in results], label="Nested CV ROC-AUC",
           color="#4C78A8", capsize=3)
    ax.bar(x + w / 2, [r["test"]["roc_auc"] for r in results], w, label="Hold-out test ROC-AUC",
           color="#E4572E")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Model comparison")
    ax.legend(loc="lower right")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def summary_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {"model": r["name"]}
        for m in SCORING:
            mean, std = r["nested_cv"][m]
            row[f"cv_{m}"] = f"{mean:.3f} +/- {std:.3f}"
        for m in SCORING:
            row[f"test_{m}"] = round(r["test"][m], 3)
        row["best_params"] = json.dumps(
            {k.replace("model__", ""): v for k, v in r["best_params"].items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _file_md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()  # nosec - fingerprint only


def train(df: pd.DataFrame | None = None, fast: bool = False, use_mlflow: bool = True,
          save: bool = True, model_path: Path | str = config.MODEL_PATH,
          metadata_path: Path | str = config.METADATA_PATH, data_path: Path | str = config.DATA_CLEAN,
          n_jobs: int = -1) -> dict:
    df = data.load_clean(data_path) if df is None else df
    X_tr, X_te, y_tr, y_te = split(df)
    cands = candidates(fast)
    log.info("train=%d test=%d candidates=%s", len(X_tr), len(X_te), list(cands))

    mlflow = None
    if use_mlflow:
        import mlflow  # local import keeps tests/serving light
        import mlflow.sklearn
        from mlflow.models import infer_signature
        mlflow.set_tracking_uri(config.MLFLOW_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    data_hash = _file_md5(Path(data_path)) if Path(data_path).exists() else "n/a"
    common_params = {"seed": config.RANDOM_STATE, "test_size": config.TEST_SIZE,
                     "outer_cv_folds": 5, "inner_cv_folds": 3, "tuning_metric": "roc_auc",
                     "n_train": len(X_tr), "n_test": len(X_te), "data_md5": data_hash,
                     "n_features_raw": len(config.FEATURES), "package_version": __version__,
                     "sklearn_version": sklearn.__version__}
    results, run_ids = [], {}
    tmp = Path(tempfile.mkdtemp(prefix="heart_artifacts_"))

    parent_ctx = mlflow.start_run(run_name=f"training-{datetime.now():%Y%m%d-%H%M%S}") if mlflow else None
    try:
        if mlflow:
            mlflow.log_params(common_params)
            mlflow.set_tag("stage", "model-selection")
        for name, (est, grid) in cands.items():
            log.info("== %s", name)
            res = evaluate_candidate(name, est, grid, X_tr, y_tr, X_te, y_te, n_jobs=n_jobs)
            results.append(res)
            log.info("   nested CV ROC-AUC %.3f | test ROC-AUC %.3f | test acc %.3f",
                     res["nested_cv"]["roc_auc"][0], res["test"]["roc_auc"], res["test"]["accuracy"])
            if mlflow:
                art = tmp / name
                plots = make_plots(res, X_te, y_te, art)
                with mlflow.start_run(run_name=name, nested=True) as child:
                    run_ids[name] = child.info.run_id
                    mlflow.log_params(common_params)
                    mlflow.log_param("model", name)
                    mlflow.log_params(
                        {k.replace("model__", "best_"): v for k, v in res["best_params"].items()})
                    mlflow.log_metric("tuning_cv_roc_auc", res["tuning_cv_roc_auc"])
                    for m, (mean, std) in res["nested_cv"].items():
                        mlflow.log_metric(f"cv_{m}_mean", mean)
                        mlflow.log_metric(f"cv_{m}_std", std)
                    for m, v in res["test"].items():
                        mlflow.log_metric(f"test_{m}", v)
                    for p in plots:
                        mlflow.log_artifact(str(p), "plots")
                    res["cv_results"].to_csv(art / "grid_search_cv_results.csv", index=False)
                    mlflow.log_artifact(str(art / "grid_search_cv_results.csv"), "tuning")
                    rep = classification_report(y_te, res["pipeline"].predict(X_te),
                                                target_names=["no disease", "disease"])
                    (art / "classification_report.txt").write_text(rep)
                    mlflow.log_artifact(str(art / "classification_report.txt"), "reports")
                    sig = infer_signature(X_te, res["pipeline"].predict_proba(X_te)[:, 1])
                    mlflow.sklearn.log_model(res["pipeline"], name="model", signature=sig,
                                             input_example=X_te.head(3),
                                             serialization_format="cloudpickle")

        best = select_model(results)
        table = summary_table(results)
        log.info("\n%s", table.drop(columns=["best_params"]).to_string(index=False))
        log.info("best model: %s", best["name"])

        if mlflow:
            table.to_csv(tmp / "model_comparison.csv", index=False)
            mlflow.log_artifact(str(tmp / "model_comparison.csv"))
            mlflow.log_artifact(str(comparison_plot(results, tmp / "model_comparison.png")))
            mlflow.log_param("selected_model", best["name"])
            mlflow.log_param("selection_rule", "one-standard-error on nested-CV ROC-AUC, prefer simpler")
            mlflow.log_metric("selected_cv_roc_auc", best["nested_cv"]["roc_auc"][0])
            mlflow.log_metric("selected_test_roc_auc", best["test"]["roc_auc"])
            try:
                mv = mlflow.register_model(f"runs:/{run_ids[best['name']]}/model", "heart-disease-classifier")
                mlflow.set_tag("registered_model_version", str(mv.version))
            except Exception as exc:  # noqa: BLE001 - registry is optional
                log.warning("model registry unavailable: %s", exc)
    finally:
        if parent_ctx:
            mlflow.end_run()

    metadata = {
        "model_name": best["name"],
        "model_version": f"{__version__}-{datetime.now(timezone.utc):%Y%m%d%H%M}",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": config.FEATURES, "target": config.TARGET,
        "best_params": {k: (v if v is None or isinstance(v, (int, float, str)) else str(v))
                        for k, v in best["best_params"].items()},
        "nested_cv_metrics": {m: {"mean": round(v[0], 4), "std": round(v[1], 4)}
                              for m, v in best["nested_cv"].items()},
        "test_metrics": {m: round(v, 4) for m, v in best["test"].items()},
        "data_md5": data_hash, "n_train": len(X_tr), "n_test": len(X_te),
        "python": platform.python_version(), "sklearn": sklearn.__version__,
    }
    if save:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best["pipeline"], model_path)
        Path(metadata_path).write_text(json.dumps(metadata, indent=2))
        log.info("saved model -> %s", model_path)
        if mlflow:  # also export a self-describing MLflow model directory
            import shutil
            import mlflow.sklearn
            out = Path(model_path).parent / "mlflow_model"
            shutil.rmtree(out, ignore_errors=True)
            mlflow.sklearn.save_model(best["pipeline"], str(out), serialization_format="cloudpickle")
    return {"best": best, "results": results, "table": table, "metadata": metadata}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="tiny hyper-parameter grids")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--data", type=Path, default=config.DATA_CLEAN)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not Path(args.data).exists():  # make the script work from a clean checkout
        data.prepare()
    out = train(fast=args.fast, use_mlflow=not args.no_mlflow, data_path=args.data)
    print(json.dumps(out["metadata"]["test_metrics"], indent=2))


if __name__ == "__main__":
    main()
