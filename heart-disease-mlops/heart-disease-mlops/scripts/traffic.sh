#!/usr/bin/env bash
# Generate N varied /predict calls so Prometheus/Grafana have something to show.
BASE="${1:-http://localhost:8000}"; N="${2:-60}"
for i in $(seq 1 "$N"); do
  curl -s -o /dev/null -X POST "$BASE/predict" -H 'Content-Type: application/json' -d \
  "{\"age\":$((30+i%45)),\"sex\":$((i%2)),\"cp\":$((i%4)),\"trestbps\":$((105+i%60)),\"chol\":$((170+i*3%200)),\"fbs\":0,\"restecg\":$((i%3)),\"thalach\":$((200-i%90)),\"exang\":$((i%2)),\"oldpeak\":$((i%5)).0,\"slope\":$((i%3)),\"ca\":$((i%4)),\"thal\":$((1+i%3))}"
  sleep 0.1
done; echo "sent $N requests to $BASE"
