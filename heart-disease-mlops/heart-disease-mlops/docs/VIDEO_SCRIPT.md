# 3-5 minute video: shot list

Record the screen with the terminal and a browser (OBS / QuickTime / Win+G).

1. **0:00 Intro (15 s)** - problem, dataset, repo on GitHub. Show `docs/architecture.png`.
2. **0:15 Data + EDA (30 s)** - `python scripts/download_data.py && python -m heart.data`; open `notebooks/01_eda.ipynb`, scroll to class balance, heatmap.
3. **0:45 Training + MLflow (60 s)** - `python -m heart.train --fast` (or show the saved run); `make mlflow-ui`, compare the 3 runs, open one run (params, metrics, ROC plot, model artifact). Say why logistic regression was selected (one-standard-error rule).
4. **1:45 Tests + CI (45 s)** - `pytest`; show `.github/workflows/ci.yml`; show a green run on GitHub and its artifacts. Optional: push a broken commit to show a red run.
5. **2:30 Docker (40 s)** - `docker build`, `docker run`, `bash scripts/demo_api.sh`.
6. **3:10 Kubernetes (45 s)** - `kubectl apply -k k8s/`, `kubectl -n heart-api get pods,svc`, call the service.
7. **3:55 Monitoring (40 s)** - `docker compose up`, `bash scripts/traffic.sh localhost:8000 200`, show Grafana panels and `logs/api.log`.
8. **4:35 Wrap-up (15 s)** - limitations (small dataset, not a medical device), repo link.
