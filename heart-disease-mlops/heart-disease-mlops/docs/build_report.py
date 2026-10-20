"""Build docs/Heart_Disease_MLOps_Report.pdf.   Usage: REPO_URL=https://github.com/me/repo python docs/build_report.py
Screenshots named in screenshots/README.md are embedded when present, otherwise a marked placeholder is drawn."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image as PILImage  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_JUSTIFY  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,  # noqa: E402
                                SimpleDocTemplate, Spacer, Table, TableStyle)
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay  # noqa: E402

from heart import config, data, train  # noqa: E402
from heart.features import feature_names  # noqa: E402

REPO_URL = os.getenv("REPO_URL", "https://github.com/<your-username>/heart-disease-mlops")
OUT = ROOT / "docs" / "Heart_Disease_MLOps_Report.pdf"
FIG, SHOT = ROOT / "reports" / "figures", ROOT / "screenshots"
meta = json.loads(config.METADATA_PATH.read_text())

# ---- figures derived from the saved model (test split only) ----
df = data.load_clean()
X_tr, X_te, y_tr, y_te = train.split(df)
model = joblib.load(config.MODEL_PATH)
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
RocCurveDisplay.from_estimator(model, X_te, y_te, ax=ax[0])
ax[0].plot([0, 1], [0, 1], "k--", lw=1)
ax[0].set_title("ROC curve (hold-out test)")
ConfusionMatrixDisplay.from_estimator(model, X_te, y_te, ax=ax[1], cmap="Blues",
                                      display_labels=["No disease", "Disease"], colorbar=False)
ax[1].set_title("Confusion matrix (test)")
coef = model.named_steps["model"].coef_.ravel()
names = np.array(feature_names(model.named_steps["preprocess"]))
top = np.argsort(np.abs(coef))[-12:]
ax[2].barh(names[top], coef[top], color=["#E4572E" if c > 0 else "#4C78A8" for c in coef[top]])
ax[2].set_title("Largest logistic-regression coefficients\n(red raises risk, blue lowers it)")
fig.tight_layout()
fig.savefig(FIG / "final_model_eval.png", dpi=140)
plt.close(fig)

# ---- styles ----
ss = getSampleStyleSheet()
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.6, leading=13.2, alignment=TA_JUSTIFY, spaceAfter=5)
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, spaceBefore=8, spaceAfter=6, textColor=colors.HexColor("#1f3a5f"), keepWithNext=1)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=6, spaceAfter=3, textColor=colors.HexColor("#1f3a5f"), keepWithNext=1)
CAP = ParagraphStyle("CAP", parent=B, fontSize=8.3, leading=10.5, textColor=colors.HexColor("#444"), alignment=1, spaceAfter=8)
CODE = ParagraphStyle("CODE", fontName="Courier", fontSize=7.8, leading=9.6, backColor=colors.HexColor("#f3f3f6"), borderPadding=4)
W = A4[0] - 4 * cm
S = []


def P(t, st=B):
    S.append(Paragraph(t, st))


def bullets(items):
    for i in items:
        S.append(Paragraph("&bull; " + i, ParagraphStyle("li", parent=B, leftIndent=12, firstLineIndent=-8, spaceAfter=2)))
    S.append(Spacer(1, 4))


def code(t):
    S.append(Preformatted(t.strip("\n"), CODE))
    S.append(Spacer(1, 6))


def img(path, caption, width=W, maxh=None):
    w, h = PILImage.open(path).size
    wd = width
    ht = wd * h / w
    if maxh and ht > maxh:
        wd, ht = wd * maxh / ht, maxh
    S.append(KeepTogether([Image(str(path), width=wd, height=ht), Paragraph(caption, CAP)]))


def shot(fname, caption, width=W):
    p = SHOT / fname
    if p.exists():
        img(p, caption, width)
    else:
        t = Table([[Paragraph(f"<b>SCREENSHOT TO ADD:</b> {caption}<br/><font size=7.5>save as screenshots/{fname}, "
                              f"then re-run docs/build_report.py</font>", B)]], colWidths=[W])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF3B0")),
                               ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#b59b00")),
                               ("TOPPADDING", (0, 0), (-1, -1), 14), ("BOTTOMPADDING", (0, 0), (-1, -1), 14)]))
        S.append(t)
        S.append(Spacer(1, 8))


def table(rows, widths, header=True, fs=8.3):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    st = [("FONTSIZE", (0, 0), (-1, -1), fs), ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
          ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe8f3")), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(st))
    S.append(t)
    S.append(Spacer(1, 7))


def cell(t):
    return Paragraph(t, ParagraphStyle("c", parent=B, fontSize=8.2, leading=10, alignment=0, spaceAfter=0))


# ================= PAGE 1: title + intro =================
S.append(Spacer(1, 1.2 * cm))
S.append(Paragraph("Heart Disease Risk Prediction<br/>End-to-End MLOps Pipeline",
                   ParagraphStyle("T", parent=ss["Title"], fontSize=24, leading=30, textColor=colors.HexColor("#1f3a5f"))))
S.append(Paragraph("Machine Learning Operations (MLOps) AIMLCZG523 - Assignment 01",
                   ParagraphStyle("st", parent=B, alignment=1, fontSize=11)))
S.append(Paragraph("Student: <b>[your name / ID]</b> &nbsp;|&nbsp; Repository: " + REPO_URL,
                   ParagraphStyle("st2", parent=B, alignment=1, fontSize=9.5)))
S.append(Spacer(1, 0.5 * cm))
P("1. Introduction", H1)
P("This project builds a classifier that predicts the presence of heart disease from 13 routine clinical measurements "
  "(UCI Heart Disease, Cleveland subset) and delivers it the way a production team would: reproducible data and training code, "
  "tracked experiments, automated tests and CI, a Docker image, Kubernetes manifests and Prometheus/Grafana monitoring. "
  "The emphasis is on the engineering around the model rather than on squeezing the last point of accuracy out of 302 patients.")
P("<b>Key results.</b> The selected model is a logistic-regression pipeline with a hold-out ROC-AUC of "
  f"<b>{meta['test_metrics']['roc_auc']:.3f}</b> (accuracy {meta['test_metrics']['accuracy']:.3f}, precision "
  f"{meta['test_metrics']['precision']:.3f}, recall {meta['test_metrics']['recall']:.3f}) and a nested cross-validated ROC-AUC of "
  f"{meta['nested_cv_metrics']['roc_auc']['mean']:.3f} +/- {meta['nested_cv_metrics']['roc_auc']['std']:.3f}. "
  "The service passes 38 automated tests (78% line coverage), reproduces from a fresh virtual environment using only "
  "<font face='Courier'>requirements-dev.txt</font>, and exposes <font face='Courier'>/predict</font> with prediction and confidence.")
P("<b>Honest scope note.</b> The model is for education, not clinical use; the test set has only 61 patients so reported test "
  "metrics have wide uncertainty (roughly +/- 0.1 on accuracy).")
img(ROOT / "docs" / "architecture.png", "Figure 1 - End-to-end architecture: data, development/tracking, CI/CD and serving/monitoring.")

# ================= setup =================
P("2. Setup and installation", H1)
code(f"""git clone {REPO_URL}.git && cd heart-disease-mlops
python3 -m venv .venv && source .venv/bin/activate        # Python 3.12
pip install -r requirements-dev.txt
export PYTHONPATH=src
python scripts/download_data.py      # data/raw/heart.csv (copy committed)
python -m heart.data                 # cleaned dataset -> data/processed/
python -m heart.train                # tune, evaluate, log to MLflow, save models/
pytest --cov=src/heart && flake8 src tests scripts
uvicorn heart.api:app --port 8000    # or: docker build -t heart-disease-api:1.0.0 . """)
P("Dependencies are pinned in <font face='Courier'>requirements.txt</font> (training) and "
  "<font face='Courier'>requirements-serve.txt</font> (slim inference image without MLflow/plotting libraries). "
  "To verify reproducibility, a pristine copy of the repository without any generated artefacts was installed into a new virtualenv "
  "and the whole flow (download, clean, train, test) was executed successfully (Figure 2).")
shot("01_terminal_clean_setup_run.png", "Figure 2 - Terminal output of the clean-environment run (real output; text rendered as an image).", W)

# ================= data + EDA =================
P("3. Data acquisition, cleaning and EDA", H1)
P("<b>Source.</b> The UCI Machine Learning Repository host was not reachable from the build environment, so "
  "<font face='Courier'>scripts/download_data.py</font> downloads the widely mirrored 14-column CSV of the same Cleveland data from GitHub "
  "(with a second mirror as fallback and content validation). The README explains the official UCI route and its different coding.")
P("<b>Cleaning</b> (<font face='Courier'>heart.data.clean</font>): (i) schema validation; (ii) numeric coercion; (iii) invalid category codes "
  "(<font face='Courier'>ca = 4</font>, <font face='Courier'>thal = 0</font>) become missing; (iv) one exact duplicate removed; "
  "(v) <b>target inversion</b>. In this mirror <font face='Courier'>target = 1</font> encodes <i>absence</i> of disease: those patients have a "
  "higher maximum heart rate and less exercise angina, and the class counts match UCI's 164 healthy / 139 diseased. After cleaning, "
  "<font face='Courier'>1 = disease present</font>. Result: 302 patients, 164 negative / 138 positive, and 6 missing cells (4 in ca, 2 in thal) - "
  "exactly UCI's documented missing values. Missing values are imputed inside the model pipeline so that statistics come only from training folds.")
table([["Feature group", "Columns", "Treatment"],
       [cell("Continuous"), cell("age, trestbps, chol, thalach, oldpeak"), cell("median imputation, standard scaling")],
       [cell("Engineered"), cell("pct_max_hr = thalach/(220-age); st_depression_with_angina = oldpeak x exang"), cell("as continuous")],
       [cell("Categorical"), cell("sex, cp, fbs, restecg, exang, slope, ca, thal"), cell("most-frequent imputation, one-hot (unknown ignored)")]],
      [3.2 * cm, 8.3 * cm, W - 11.5 * cm])
img(FIG / "class_balance.png", "Figure 3 - Class balance (54% / 46%): near-balanced, so accuracy is meaningful; recall and ROC-AUC are reported too.", W * 0.5)
img(FIG / "histograms.png", "Figure 4 - Continuous features by class. Disease cases are older, reach a lower maximum heart rate and show larger ST depression.", W)
img(FIG / "correlation_heatmap.png", "Figure 5 - Correlation heatmap. Strongest links to the target: ca (+0.46), exang (+0.44), cp (-0.43), oldpeak (+0.43), thalach (-0.42); "
    "no predictor pair above |r| = 0.58, so no feature was dropped.", W * 0.78)
P("<b>Other observations.</b> <font face='Courier'>cp = 0</font> covers 47% of patients and 73% of them have disease (18-30% for other codes); "
  "exercise angina raises the disease rate from 31% to 77%. Cholesterol above 400 mg/dl and oldpeak above 5 are rare but plausible, so they were kept "
  "(scaling and the chosen models tolerate them).")

# ================= modelling =================
P("4. Feature engineering and model development", H1)
P("All preprocessing and the estimator live in a single scikit-learn <font face='Courier'>Pipeline</font>, guaranteeing identical transformations "
  "in training, cross-validation, notebooks, the API and the container. Data were split 80/20 (stratified, seed 42): 241 training and 61 test patients. "
  "Three classifiers were compared, each tuned with grid search (5-fold stratified, ROC-AUC):")
table([["Model", "Hyper-parameter grid"],
       [cell("Logistic regression"), cell("C in {0.01, 0.1, 0.5, 1, 5, 10}; class_weight in {None, balanced}  (12 combinations)")],
       [cell("Random forest"), cell("n_estimators {100, 300}; max_depth {3, 5, None}; min_samples_leaf {1, 3, 5}  (18)")],
       [cell("Gradient boosting"), cell("n_estimators {50, 100, 200}; learning_rate {0.03, 0.1}; max_depth {2, 3}  (12)")]],
      [4 * cm, W - 4 * cm])
P("<b>Evaluation protocol.</b> To avoid optimistic estimates from tuning on the same folds, performance was estimated with <i>nested</i> cross-validation "
  "(3-fold inner search inside a 5-fold outer loop) on the training split using accuracy, precision, recall, F1 and ROC-AUC; each tuned model was then scored "
  "once on the hold-out test set.")
rows = [["Model", "CV ROC-AUC", "CV accuracy", "CV recall", "Test ROC-AUC", "Test acc.", "Test recall"],
        ["Logistic regression (selected)", "0.907 +/- 0.039", "0.846", "0.809", "0.918", "0.836", "0.786"],
        ["Random forest", "0.910 +/- 0.036", "0.830", "0.791", "0.894", "0.787", "0.679"],
        ["Gradient boosting", "0.900 +/- 0.030", "0.830", "0.809", "0.879", "0.770", "0.643"]]
table(rows, [4.6 * cm, 2.7 * cm, 2.1 * cm, 1.9 * cm, 2.4 * cm, 1.7 * cm, W - 15.4 * cm])
P("<b>Model selection.</b> The three CV ROC-AUC values differ by 0.01, well within the fold-to-fold standard deviation (0.03-0.04), so choosing the "
  "nominal winner (random forest, 0.910) would mean choosing noise. I therefore applied the <b>one-standard-error rule</b>: among models within one SD of the best "
  "CV score, take the simplest. This rule uses CV results only - the test set played no part in the decision. Logistic regression is selected; it is also "
  "the most interpretable and had the best CV accuracy and F1. Best parameters and all metrics are stored in <font face='Courier'>models/metadata.json</font>.")
img(FIG / "final_model_eval.png", "Figure 6 - Selected model on the hold-out test set: ROC curve, confusion matrix and the largest coefficients.", W)
P("<b>Limitations.</b> With 61 test patients, one extra error moves accuracy by 1.6 points. The default 0.5 threshold gives recall 0.79; for screening a lower "
  "threshold could be configured (notebook 02 shows the precision/recall trade-off).")
shot("02_terminal_training_results.png", "Figure 7 - Output of python -m heart.train (nested CV and test metrics for all candidates).", W)

# ================= tracking + packaging =================
P("5. Experiment tracking (MLflow)", H1)
P("<font face='Courier'>heart.train</font> opens one parent run and one child run per model in the experiment <i>heart-disease-risk</i> "
  "(SQLite backend <font face='Courier'>mlflow.db</font>). Each child run logs: parameters (model, best hyper-parameters, seed, split sizes, CV folds, data MD5, "
  "library versions); metrics (tuning score, nested-CV mean and SD for five metrics, five test metrics); artifacts (ROC curve, confusion matrix, feature "
  "importance, full grid-search results CSV, classification report) and the fitted pipeline with a signature and input example. The parent run stores the comparison "
  "table and chart, the selection rule and the selected model, which is registered as <i>heart-disease-classifier</i> in the MLflow model registry.")
shot("03_terminal_mlflow_runs.png", "Figure 8 - MLflow runs queried via the Python API (real output).", W)
shot("15_mlflow_ui_runs.png", "Figure 9 - MLflow UI: experiment table comparing the three runs.")
shot("16_mlflow_ui_run_detail.png", "Figure 10 - MLflow UI: parameters, metrics and artifacts of one run.")

P("6. Model packaging and reproducibility", H1)
bullets(["<b>Formats:</b> <font face='Courier'>models/heart_pipeline.joblib</font> (used by the API) and an MLflow model directory "
         "<font face='Courier'>models/mlflow_model/</font> (MLmodel, conda.yaml, requirements.txt). The serialized object is the <i>whole</i> pipeline "
         "(feature engineering, imputers, scaler, encoder, classifier), so no preprocessing code is duplicated at serving time.",
         "<b>Environment:</b> pinned <font face='Courier'>requirements*.txt</font>; Python 3.12; fixed random seeds; training verified deterministic by a unit test.",
         "<b>Lineage:</b> <font face='Courier'>metadata.json</font> records model version, training time, metrics, feature list, library versions and the MD5 of the training data.",
         "<b>Clean-setup proof:</b> Figure 2 (fresh venv, requirements only, 38 tests passing)."])

# ================= CI/CD =================
P("7. CI/CD pipeline and automated testing", H1)
P("<b>Tests</b> (pytest, 38 tests, 78% coverage): data cleaning (invalid codes, duplicates, target inversion, published-statistics check), feature engineering "
  "(values, immutability, finite output with missing data, unseen categories, standardisation), model training (metric ranges, determinism, artifact round-trip, "
  "optional-field handling, risk monotonicity, selection rule), the download script (mocked network and mirror fallback), EDA output and the API (valid/invalid inputs, "
  "metrics endpoint, request-ID correlation, no patient payload in logs).")
shot("04_terminal_pytest_coverage.png", "Figure 11 - pytest with coverage (real output).", W)
P("<b>Workflow</b> (<font face='Courier'>.github/workflows/ci.yml</font>): <i>lint</i> (flake8) &rarr; <i>test</i> (pytest, coverage, JUnit XML uploaded as artifact) &rarr; "
  "<i>train</i> (download, clean, train with MLflow, quality gate: hold-out ROC-AUC must be at least 0.80; model, <font face='Courier'>mlflow.db</font> and training log uploaded) "
  "&rarr; <i>docker</i> (build the image, run it, wait for /health, POST a patient to /predict, check /metrics; container logs uploaded) &rarr; <i>publish</i> "
  "(push image to GHCR on main only). Jobs depend on each other with <font face='Courier'>needs</font>, so any non-zero exit stops the pipeline and the "
  "step log shows the failure. To demonstrate this, a deliberate bug was introduced into the feature code in a throw-away copy: the test step failed and returned exit code 1 (Figure 12).")
shot("06_terminal_ci_failure_demo.png", "Figure 12 - A broken commit makes the test step fail with a non-zero exit code (real output from a local simulation).", W)
shot("10_github_actions_run.png", "Figure 13 - GitHub Actions: successful run with all jobs.")
shot("11_github_actions_artifacts.png", "Figure 14 - Artifacts uploaded by the run (test reports, model and tracking DB, container logs).")

# ================= container =================
P("8. Containerisation", H1)
P("The Dockerfile uses <font face='Courier'>python:3.12-slim</font>, installs the slim serving requirements first (layer caching), copies only "
  "<font face='Courier'>src/</font> and the trained model artifacts, runs as an unprivileged user (UID 10001), declares a HEALTHCHECK and starts uvicorn on port 8000. "
  "<font face='Courier'>.dockerignore</font> keeps data, notebooks, tests and MLflow files out of the image.")
table([["Endpoint", "Purpose"],
       [cell("POST /predict"), cell("JSON with 13 features (ca, thal optional) &rarr; prediction, label, confidence (probability of the predicted class), probability_disease, model_version; invalid input &rarr; HTTP 422")],
       [cell("GET /health"), cell("Liveness/readiness, model version")],
       [cell("GET /model-info"), cell("Stored metadata: metrics, parameters, data hash")],
       [cell("GET /metrics"), cell("Prometheus metrics")]], [3.3 * cm, W - 3.3 * cm])
shot("07_terminal_api_curl_demo.png", "Figure 15 - Live API test with curl: two predictions and a rejected invalid request (real output from the local uvicorn server).", W)
shot("12_docker_build_run.png", "Figure 16 - docker build and docker run followed by the same curl tests.")

# ================= deployment =================
P("9. Production deployment (Kubernetes)", H1)
P("<font face='Courier'>k8s/</font> contains a Namespace, a Deployment (2 replicas, rolling update with zero unavailable, readiness and liveness probes on "
  "/health, CPU/memory requests and limits, non-root security context, Prometheus scrape annotations), a Service of type <b>LoadBalancer</b> (port 80 &rarr; 8000), "
  "an optional Ingress and a HorizontalPodAutoscaler (2-5 pods at 70% CPU), plus a kustomization file. Deploy with <font face='Courier'>kubectl apply -k k8s/</font>. "
  "On Minikube the image is loaded locally and <font face='Courier'>minikube tunnel</font> provides the external IP; on a cloud cluster the image is pulled from GHCR "
  "(published by the pipeline) after editing the <font face='Courier'>image:</font> line.")
shot("13_kubectl_get_pods_svc.png", "Figure 17 - kubectl get pods,svc in namespace heart-api (pods Running, service external IP).")
shot("14_k8s_api_call.png", "Figure 18 - Calling /health and /predict through the Kubernetes service.")

# ================= monitoring =================
P("10. Monitoring and logging", H1)
P("<b>Logging.</b> Every request produces one JSON log line (timestamp, request ID, method, path, status, latency, client) on stdout and in a rotating file "
  "<font face='Courier'>logs/api.log</font>; each prediction adds a line with the same request ID, the predicted class and confidence. Raw patient payloads are deliberately "
  "<b>not</b> logged because they are health data (a unit test enforces this).")
shot("09_terminal_api_request_logs.png", "Figure 19 - Structured request and prediction logs sharing a request_id (real output).", W)
P("<b>Metrics.</b> Prometheus counters and histograms: <font face='Courier'>api_requests_total</font> (method, endpoint, status), "
  "<font face='Courier'>api_request_latency_seconds</font>, <font face='Courier'>predictions_total</font> (by class), "
  "<font face='Courier'>prediction_confidence</font> and <font face='Courier'>model_info</font>. "
  "<font face='Courier'>docker compose up</font> starts the API, Prometheus and Grafana with a provisioned dashboard (request rate, p95 latency, predictions by class, "
  "error count, median confidence). Prediction-class and confidence drift would be visible there; true label feedback is not available, so accuracy cannot be monitored online.")
shot("08_terminal_api_metrics.png", "Figure 20 - /metrics output after test traffic (real output).", W)
shot("17_prometheus_targets.png", "Figure 21 - Prometheus target heart-api is UP.")
shot("18_grafana_dashboard.png", "Figure 22 - Grafana dashboard under generated load.")

# ================= wrap-up =================
P("11. Conclusions, limitations and next steps", H1)
bullets(["A simple, interpretable model performs as well as tree ensembles on this small tabular dataset; the one-standard-error rule documents why it was chosen.",
         "The dataset (302 patients, single site) limits generalisation and the 61-patient test set limits precision of the estimates; the service must not be used clinically.",
         "Not exercised in the build environment: Docker image build, Kubernetes deployment and GitHub Actions execution (configuration was validated for syntax only; evidence screenshots are added by the author).",
         "Next steps: data-drift monitoring on input features, a model-registry-driven rollout, a configurable decision threshold for screening, calibration analysis and a larger multi-site dataset."])
P("<b>Code repository:</b> " + REPO_URL)
P("<b>Deployed API:</b> not publicly hosted; run locally with Docker or Minikube as described in Sections 2, 8 and 9.")

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                        title="Heart Disease MLOps Report", author="[student]")


def footer(c, d):
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.grey)
    c.drawCentredString(A4[0] / 2, 1 * cm, f"MLOps Assignment 01 - Heart Disease Risk - page {d.page}")


doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
