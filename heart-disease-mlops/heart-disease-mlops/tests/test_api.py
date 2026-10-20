import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tiny_model, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tiny_model["path"]))
    monkeypatch.setenv("METADATA_PATH", str(tiny_model["meta"]))
    from heart.api import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_returns_prediction_and_confidence(client, sample):
    r = client.post("/predict", json=sample)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert 0 <= body["probability_disease"] <= 1
    assert body["model_version"]
    assert "x-request-id" in r.headers


def test_optional_fields_may_be_omitted(client, sample):
    sample.pop("ca"), sample.pop("thal")
    assert client.post("/predict", json=sample).status_code == 200


@pytest.mark.parametrize("field,value", [("age", -5), ("sex", 2), ("cp", 7), ("oldpeak", "high"),
                                         ("chol", 5000)])
def test_invalid_input_is_rejected(client, sample, field, value):
    r = client.post("/predict", json=dict(sample, **{field: value}))
    assert r.status_code == 422


def test_missing_required_field_is_rejected(client, sample):
    sample.pop("age")
    assert client.post("/predict", json=sample).status_code == 422


def test_model_info_exposes_metrics(client):
    assert "test_metrics" in client.get("/model-info").json()


def test_metrics_endpoint_counts_requests(client, sample):
    client.post("/predict", json=sample)
    text = client.get("/metrics").text
    assert 'api_requests_total{endpoint="/predict",method="POST",status="200"}' in text
    assert "predictions_total" in text and "api_request_latency_seconds_bucket" in text


def test_request_log_excludes_patient_payload(client, sample, caplog):
    import logging
    with caplog.at_level(logging.INFO, logger="heart.api"):
        client.post("/predict", json=sample)
    logged = " ".join(r.getMessage() + str(getattr(r, "extra_fields", "")) for r in caplog.records)
    assert "prediction" in logged and "233" not in logged     # chol value not logged


def test_prediction_and_request_logs_share_request_id(client, sample, caplog):
    import logging
    with caplog.at_level(logging.INFO, logger="heart.api"):
        r = client.post("/predict", json=sample, headers={"x-request-id": "abc123"})
    ids = [getattr(rec, "extra_fields", {}).get("request_id") for rec in caplog.records]
    assert r.headers["x-request-id"] == "abc123"
    assert ids.count("abc123") == 2          # one "prediction" line + one "request" line
