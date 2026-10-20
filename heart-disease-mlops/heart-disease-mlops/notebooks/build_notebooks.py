"""Generate and execute the three notebooks:  python notebooks/build_notebooks.py"""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).resolve().parent
SETUP = '''import sys, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
%matplotlib inline
pd.set_option("display.width", 140); pd.set_option("display.max_columns", 30)
from heart import config, data'''


def nb(cells, name):
    book = nbf.v4.new_notebook()
    book.cells = [nbf.v4.new_markdown_cell(c) if k == "md" else nbf.v4.new_code_cell(c) for k, c in cells]
    book.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(book, {"metadata": {"path": str(HERE)}})
    nbf.write(book, HERE / name)
    print("wrote", name)


# ---------------------------------------------------------------- 01 EDA
nb([
 ("md", "# 01 - Data acquisition, cleaning and EDA\nHeart Disease (Cleveland / UCI) - 13 clinical features + binary target."),
 ("code", SETUP + "\nfrom heart import eda\nplt.switch_backend('module://matplotlib_inline.backend_inline')  # heart.* forces Agg for CLI/CI; restore inline"),
 ("md", "## 1. Acquire the data\n`scripts/download_data.py` fetches the CSV (see README for the official UCI route). "
        "The file is already committed in `data/raw/`, so this notebook works offline."),
 ("code", "raw = data.load_raw()\nprint(raw.shape)\nraw.head()"),
 ("code", "pd.Series(config.DATA_DICTIONARY, name='meaning').to_frame()"),
 ("md", "## 2. Quality checks on the raw file"),
 ("code", "print('duplicates:', raw.duplicated().sum())\nprint('NaN cells:', int(raw.isna().sum().sum()))\n"
         "print('ca value counts   :', raw.ca.value_counts().sort_index().to_dict())\n"
         "print('thal value counts :', raw.thal.value_counts().sort_index().to_dict())\n"
         "print('target value counts:', raw.target.value_counts().to_dict())"),
 ("md", "**Findings**\n* `ca` has 5 rows coded `4` and `thal` has 2 rows coded `0` - both outside the data dictionary; these are placeholders for *missing*.\n"
        "* One exact duplicate row.\n"
        "* In this public mirror `target == 1` means **no disease**. Check: patients with `target == 1` have a *higher* maximum heart rate "
        "and *less* exercise angina - the opposite of what disease would cause. The counts (164 healthy / 139 diseased in the original) "
        "also match the inverted label. `clean()` therefore flips it so that **1 = disease present**."),
 ("code", "print(raw.groupby('target')[['thalach','exang','oldpeak','ca']].mean().round(2))"),
 ("md", "## 3. Clean"),
 ("code", "df = data.prepare()\nprint(df.shape)\nprint('missing per column:', df.isna().sum()[df.isna().sum() > 0].to_dict())\ndf.describe().T.round(2)"),
 ("md", "Missing values (6 cells) are **not** filled here. They are imputed inside the model pipeline "
        "(median / most-frequent), so the statistics come only from training folds and nothing leaks into validation data."),
 ("md", "## 4. Class balance"),
 ("code", "eda.class_balance(df); plt.show()"),
 ("md", "Classes are nearly balanced (54% / 46%), so accuracy is meaningful, but we still report precision, recall and ROC-AUC."),
 ("md", "## 5. Distributions"),
 ("code", "eda.histograms(df); plt.show()"),
 ("code", "eda.categorical_counts(df); plt.show()"),
 ("md", "* Disease cases are older, reach a lower maximum heart rate (`thalach`) and have larger ST depression (`oldpeak`).\n"
        "* Strong categorical signals: `cp = 0` (47% of patients; 73% of them have disease vs 18-30% for the other codes), "
        "`exang = 1` (77% vs 31%), `ca >= 1` (68-85% vs 26%) and `thal` 1 or 3 (67-76% vs 22% for code 2)."),
 ("md", "## 6. Correlations"),
 ("code", "eda.correlation_heatmap(df); plt.show()"),
 ("code", "df.corr()['target'].drop('target').sort_values(key=abs, ascending=False).round(3).to_frame('corr_with_target')"),
 ("md", "Strongest correlations with the target: `ca` (+0.46), `exang` (+0.44), `cp` (-0.43), `oldpeak` (+0.43), `thalach` (-0.42). "
        "No pair of predictors is extremely collinear (max |r| = 0.58, `oldpeak` vs `slope`), so no feature needs to be dropped."),
 ("md", "## 7. Outliers"),
 ("code", "eda.boxplots(df); plt.show()"),
 ("md", "A few high cholesterol readings (> 400 mg/dl) and `oldpeak` values above 5 exist but are clinically plausible, so they are kept. "
        "Standard scaling plus tree-based models are tolerant to them."),
 ("code", "eda.run_all(df)  # also writes every figure to reports/figures/ for the report\nsorted(p.name for p in config.FIGURES_DIR.glob('*.png'))"),
], "01_eda.ipynb")

# ---------------------------------------------------------------- 02 training
nb([
 ("md", "# 02 - Feature engineering, model selection and evaluation\nThe production training run is `python -m heart.train` (logged to MLflow). "
        "This notebook explains the choices and reads the results back from MLflow."),
 ("code", SETUP + "\nimport json, mlflow\nfrom sklearn.model_selection import cross_validate, StratifiedKFold\n"
         "from sklearn.linear_model import LogisticRegression\nfrom sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n"
         "from sklearn.dummy import DummyClassifier\nfrom heart import train as T\nfrom heart.features import build_preprocessor, feature_names\nplt.switch_backend('module://matplotlib_inline.backend_inline')  # heart.train forces Agg; restore inline"),
 ("code", "df = data.load_clean()\nX_tr, X_te, y_tr, y_te = T.split(df)\nprint(X_tr.shape, X_te.shape, y_tr.mean().round(3), y_te.mean().round(3))"),
 ("md", "## 1. Feature engineering / preprocessing (`heart/features.py`)\n"
        "| Group | Columns | Treatment |\n|---|---|---|\n"
        "| Continuous | age, trestbps, chol, thalach, oldpeak | median imputation, standard scaling |\n"
        "| Engineered | `pct_max_hr` = thalach / (220 - age); `st_depression_with_angina` = oldpeak x exang | same as continuous |\n"
        "| Categorical | sex, cp, fbs, restecg, exang, slope, ca, thal | most-frequent imputation, one-hot (unknown ignored) |\n\n"
        "Everything lives inside one scikit-learn `Pipeline`, so the exact same transformation is applied in training, CV, the API and the container."),
 ("code", "pre = build_preprocessor().fit(X_tr)\nZ = pre.transform(X_tr)\nprint('transformed shape:', Z.shape)\nprint(feature_names(pre))"),
 ("md", "## 2. Baseline comparison with default hyper-parameters (5-fold stratified CV, training split only)"),
 ("code", "cv = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)\n"
         "models = {'dummy (majority class)': DummyClassifier(strategy='most_frequent'),\n"
         "          'logistic_regression': LogisticRegression(max_iter=2000),\n"
         "          'random_forest': RandomForestClassifier(random_state=42),\n"
         "          'gradient_boosting': GradientBoostingClassifier(random_state=42)}\n"
         "rows = {}\nfor name, est in models.items():\n"
         "    r = cross_validate(T.make_pipeline(est), X_tr, y_tr, cv=cv, scoring=T.SCORING)\n"
         "    rows[name] = {m: f\"{r['test_'+m].mean():.3f} +/- {r['test_'+m].std():.3f}\" for m in T.SCORING}\n"
         "pd.DataFrame(rows).T"),
 ("md", "The dummy model confirms the scale of the task (ROC-AUC 0.5). All real models are clearly better; logistic regression is already competitive, "
        "which is typical for a small, mostly-linear clinical dataset."),
 ("md", "## 3. Hyper-parameter tuning and final selection\nGrid search (5-fold, ROC-AUC) inside a *nested* CV loop gives an estimate that includes the tuning step. "
        "The tuned models were then scored once on the hold-out test set. Results come straight from MLflow:"),
 ("code", "mlflow.set_tracking_uri(config.MLFLOW_URI)\nexp = mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT)\n"
         "runs = mlflow.search_runs([exp.experiment_id], filter_string=\"tags.mlflow.parentRunId != ''\", order_by=['start_time DESC'])\n"
         "runs = runs.drop_duplicates('params.model')\n"
         "cols = ['params.model'] + [c for c in runs.columns if c.startswith('metrics.cv_') and c.endswith('_mean')] + [c for c in runs.columns if c.startswith('metrics.test_')]\n"
         "runs[cols].rename(columns=lambda c: c.replace('metrics.','').replace('params.','')).set_index('model').round(3)"),
 ("code", "print(json.dumps(json.loads((config.METADATA_PATH).read_text()), indent=2)[:1500])"),
 ("md", "## 4. Evaluation of the selected model on the hold-out test set"),
 ("code", "import joblib\nfrom sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay, classification_report, precision_recall_curve\n"
         "model = joblib.load(config.MODEL_PATH)\nfig, ax = plt.subplots(1, 2, figsize=(10, 4))\n"
         "RocCurveDisplay.from_estimator(model, X_te, y_te, ax=ax[0]); ax[0].plot([0,1],[0,1],'k--'); ax[0].set_title('ROC (test)')\n"
         "ConfusionMatrixDisplay.from_estimator(model, X_te, y_te, ax=ax[1], cmap='Blues', display_labels=['No disease','Disease']); ax[1].set_title('Confusion matrix (test)')\nplt.tight_layout(); plt.show()\n"
         "print(classification_report(y_te, model.predict(X_te), target_names=['no disease','disease']))"),
 ("md", "## 5. Operating point\nFor screening, a missed case (false negative) is costlier than a false alarm. The curve shows what lowering the threshold would buy."),
 ("code", "p = model.predict_proba(X_te)[:, 1]\nprec, rec, thr = precision_recall_curve(y_te, p)\n"
         "plt.figure(figsize=(6,4)); plt.plot(thr, prec[:-1], label='precision'); plt.plot(thr, rec[:-1], label='recall')\n"
         "plt.axvline(0.5, color='gray', ls='--'); plt.xlabel('decision threshold'); plt.legend(); plt.title('Precision / recall vs threshold'); plt.show()\n"
         "for t in (0.3, 0.4, 0.5):\n    pred = (p >= t).astype(int)\n    print(f't={t}: precision={(pred[y_te==1].sum()/max(pred.sum(),1)):.3f} recall={pred[y_te==1].mean():.3f}')"),
 ("md", "The service uses 0.5 by default; a deployment for screening could expose the threshold as configuration."),
], "02_model_training.ipynb")

# ---------------------------------------------------------------- 03 inference
nb([
 ("md", "# 03 - Inference\nLoad the saved pipeline, score patients, and call the API in-process (same code path as the container)."),
 ("code", SETUP + "\nimport json\nfrom heart.predict import load_model, load_metadata, predict_records"),
 ("code", "model = load_model()\nmeta = load_metadata()\nprint(meta['model_name'], meta['model_version'])\nprint('test metrics:', meta['test_metrics'])"),
 ("md", "## Score two example patients"),
 ("code", "patients = [\n"
         "  dict(age=63, sex=1, cp=3, trestbps=145, chol=233, fbs=1, restecg=0, thalach=150, exang=0, oldpeak=2.3, slope=0, ca=0, thal=1),\n"
         "  dict(age=64, sex=1, cp=0, trestbps=160, chol=300, fbs=0, restecg=1, thalach=105, exang=1, oldpeak=3.5, slope=1, ca=3, thal=3),\n"
         "  dict(age=35, sex=0, cp=2, trestbps=115, chol=190, fbs=0, restecg=1, thalach=185, exang=0, oldpeak=0.0, slope=2, ca=None, thal=None)]\n"
         "pd.DataFrame(predict_records(model, patients))"),
 ("md", "The third record leaves `ca` and `thal` empty; the pipeline's imputers handle it."),
 ("md", "## Call the REST API (in-process with FastAPI's TestClient)"),
 ("code", "from fastapi.testclient import TestClient\nfrom heart.api import app\n"
         "with TestClient(app) as client:\n    print(client.get('/health').json())\n"
         "    r = client.post('/predict', json=patients[0]); print(r.status_code, json.dumps(r.json(), indent=2))\n"
         "    bad = client.post('/predict', json=dict(patients[0], age=-3)); print('invalid input ->', bad.status_code)"),
 ("md", "Equivalent `curl` once the server or container is running:\n```bash\ncurl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' \\\n"
        "  -d '{\"age\":63,\"sex\":1,\"cp\":3,\"trestbps\":145,\"chol\":233,\"fbs\":1,\"restecg\":0,\"thalach\":150,\"exang\":0,\"oldpeak\":2.3,\"slope\":0,\"ca\":0,\"thal\":1}'\n```"),
 ("md", "## Distribution of predicted risk on the whole dataset"),
 ("code", "d = data.load_clean(); probs = [r['probability_disease'] for r in predict_records(model, d[config.FEATURES].to_dict('records'))]\n"
         "sns.histplot(x=probs, hue=d.target.map({0:'no disease',1:'disease'}), bins=20, element='step'); plt.xlabel('predicted P(disease)'); plt.title('Predicted risk by true class (in-sample, illustrative)'); plt.show()"),
], "03_inference.ipynb")
