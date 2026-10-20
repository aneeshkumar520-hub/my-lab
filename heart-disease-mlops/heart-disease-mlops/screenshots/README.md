# Screenshots

`01_...09_terminal_*.png` are **images of real terminal output** captured while building the project
(`scripts/make_terminal_images.py` regenerates them from `logs/`). They are not GUI screenshots.

Capture the following yourself on your machine and save them with these exact names; `docs/build_report.py`
picks them up automatically (`python docs/build_report.py` rebuilds the PDF):

| File name | What to capture |
|---|---|
| `10_github_actions_run.png` | Actions tab: a green run showing all jobs (lint, test, train, docker) |
| `11_github_actions_artifacts.png` | Bottom of that run: the uploaded artifacts |
| `12_docker_build_run.png` | Terminal: `docker build ...`, `docker run ...`, then `bash scripts/demo_api.sh` |
| `13_kubectl_get_pods_svc.png` | `kubectl -n heart-api get pods,svc` (pods Running, service EXTERNAL-IP) |
| `14_k8s_api_call.png` | `bash scripts/demo_api.sh http://<EXTERNAL-IP>` against the cluster |
| `15_mlflow_ui_runs.png` | MLflow UI (`make mlflow-ui`): experiment table comparing the three runs |
| `16_mlflow_ui_run_detail.png` | One run's page: params, metrics, artifacts/plots |
| `17_prometheus_targets.png` | Prometheus -> Status -> Targets showing `heart-api` UP |
| `18_grafana_dashboard.png` | Grafana "Heart Disease API" dashboard after `scripts/traffic.sh` |
