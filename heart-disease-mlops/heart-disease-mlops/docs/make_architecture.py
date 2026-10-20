"""Draw docs/architecture.png (matplotlib only, no external tools needed)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent / "architecture.png"
fig, ax = plt.subplots(figsize=(13, 7.2))
ax.set_xlim(0, 13)
ax.set_ylim(0, 7.2)
ax.axis("off")

COL = {"data": "#DCE9F7", "ml": "#E4F2E1", "ci": "#FCEBD2", "run": "#F3DEEF", "mon": "#FBE3E0"}


def box(x, y, w, h, text, kind, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                                fc=COL[kind], ec="#444", lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(x1, y1, x2, y2, text="", rad=0.0, dy=0.12):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 lw=1.2, color="#333", connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, text, ha="center", fontsize=7.5, color="#333")


def lane(x, w, title):
    ax.text(x + w / 2, 6.95, title, ha="center", fontsize=10.5, fontweight="bold")
    ax.plot([x, x + w], [6.8, 6.8], color="#888", lw=0.8)


lane(0.2, 2.6, "1  Data")
lane(3.1, 3.2, "2  Develop + track")
lane(6.6, 2.9, "3  CI/CD (GitHub Actions)")
lane(9.8, 3.1, "4  Serve + monitor")

box(0.3, 5.4, 2.4, 0.9, "UCI Heart Disease\n(Cleveland, GitHub mirror)", "data")
box(0.3, 4.0, 2.4, 0.9, "scripts/download_data.py\n-> data/raw/heart.csv", "data")
box(0.3, 2.6, 2.4, 0.9, "heart.data.clean()\n-> data/processed/*.csv", "data")
box(0.3, 1.2, 2.4, 0.9, "EDA notebook +\nreports/figures", "data")
arrow(1.5, 5.4, 1.5, 4.9)
arrow(1.5, 4.0, 1.5, 3.5)
arrow(1.5, 2.6, 1.5, 2.1)

box(3.2, 5.0, 3.0, 1.3, "heart.features\nFeatureAdder + impute +\nscale + one-hot", "ml")
box(3.2, 3.35, 3.0, 1.2, "heart.train\nLogReg | RandomForest | GBM\nGridSearch + nested 5-fold CV", "ml")
box(3.2, 1.9, 3.0, 1.0, "MLflow tracking (sqlite)\nparams, metrics, plots, registry", "ml")
box(3.2, 0.4, 3.0, 1.0, "models/heart_pipeline.joblib\n+ metadata.json + mlflow_model/", "ml")
arrow(2.7, 3.05, 3.2, 4.2, "clean data", rad=-0.15)
arrow(4.7, 5.0, 4.7, 4.55)
arrow(4.7, 3.35, 4.7, 2.9)
arrow(4.7, 1.9, 4.7, 1.4)

box(6.7, 5.3, 2.7, 0.9, "git push / PR", "ci")
box(6.7, 4.1, 2.7, 0.8, "lint (flake8)", "ci")
box(6.7, 3.1, 2.7, 0.8, "pytest + coverage", "ci")
box(6.7, 2.1, 2.7, 0.8, "train + quality gate\n(ROC-AUC >= 0.80)", "ci")
box(6.7, 1.0, 2.7, 0.8, "docker build + smoke test\n-> push to GHCR", "ci")
for y1, y2 in [(5.3, 4.9), (4.1, 3.9), (3.1, 2.9), (2.1, 1.8)]:
    arrow(8.05, y1, 8.05, y2)

box(9.9, 5.2, 2.9, 1.0, "Docker image\nFastAPI /predict /health /metrics", "run")
box(9.9, 3.75, 2.9, 1.0, "Kubernetes (Minikube / GKE)\nDeployment x2, HPA, probes", "run")
box(9.9, 2.4, 2.9, 0.9, "Service: LoadBalancer / Ingress\n<- clients (curl, apps)", "run")
box(9.9, 0.4, 1.35, 1.4, "Prometheus\nscrape /metrics", "mon", fs=8.5)
box(11.45, 0.4, 1.35, 1.4, "Grafana\ndashboard", "mon", fs=8.5)
arrow(11.35, 5.2, 11.35, 4.75)
arrow(11.35, 3.75, 11.35, 3.3)
arrow(11.35, 2.4, 10.6, 1.8, "scrape", rad=0.0, dy=0.05)
arrow(11.25, 1.1, 11.45, 1.1)
arrow(9.4, 1.4, 9.9, 5.6, "image", rad=-0.25)
arrow(6.2, 0.9, 6.7, 1.4, "model", rad=0.0)

ax.set_title("Heart Disease Risk - end-to-end MLOps architecture", fontsize=13, pad=2)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("saved", OUT)
