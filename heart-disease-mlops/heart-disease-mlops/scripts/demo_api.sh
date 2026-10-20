#!/usr/bin/env bash
# Smoke-test a running API (local uvicorn, Docker container or Kubernetes service).
# Usage: scripts/demo_api.sh [BASE_URL]      default: http://localhost:8000
BASE="${1:-http://localhost:8000}"
J='Content-Type: application/json'
echo "\$ curl -s $BASE/health";                          curl -s "$BASE/health"; echo; echo
echo "\$ curl -s -X POST $BASE/predict -d @high_risk.json"
curl -s -X POST "$BASE/predict" -H "$J" -d '{"age":64,"sex":1,"cp":0,"trestbps":160,"chol":300,"fbs":0,"restecg":1,"thalach":105,"exang":1,"oldpeak":3.5,"slope":1,"ca":3,"thal":3}'; echo; echo
echo "\$ curl -s -X POST $BASE/predict -d @low_risk_no_ca_thal.json"
curl -s -X POST "$BASE/predict" -H "$J" -d '{"age":35,"sex":0,"cp":2,"trestbps":115,"chol":190,"fbs":0,"restecg":1,"thalach":185,"exang":0,"oldpeak":0.0,"slope":2}'; echo; echo
echo "\$ curl -s -w 'HTTP %{http_code}' -X POST $BASE/predict -d @invalid_age_minus3.json"
curl -s -w '\nHTTP %{http_code}\n' -X POST "$BASE/predict" -H "$J" -d '{"age":-3,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0}' | cut -c1-150
