"""FastAPI service: /predict, /health, /model-info, /metrics (Prometheus).

Logging policy: every request is logged as one JSON line (id, method, path,
status, latency). Prediction results are logged, but the raw patient payload is
NOT, because it is health data.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import (CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge,
                               Histogram, generate_latest)
from pydantic import BaseModel, ConfigDict, Field

from . import __version__, config
from .predict import load_metadata, load_model, predict_records

# ----------------------------------------------------------------- logging


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
                   "level": record.levelname, "logger": record.name,
                   "msg": record.getMessage()}
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload)


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("heart.api")
    if logger.handlers:
        return logger
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    fmt = JsonFormatter()
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    log_file = os.getenv("API_LOG_FILE")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


logger = setup_logging()


def log_event(msg: str, **fields):
    logger.info(msg, extra={"extra_fields": fields})


# ----------------------------------------------------------------- metrics
REGISTRY = CollectorRegistry()
REQUESTS = Counter("api_requests_total", "HTTP requests", ["method", "endpoint", "status"], registry=REGISTRY)
LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"], registry=REGISTRY,
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
PREDICTIONS = Counter("predictions_total", "Predictions by class", ["label"], registry=REGISTRY)
CONFIDENCE = Histogram("prediction_confidence", "Confidence of predicted class", registry=REGISTRY,
                       buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0))
MODEL_INFO = Gauge("model_info", "Loaded model (value is always 1)", ["model_name", "model_version"],
                   registry=REGISTRY)


# ----------------------------------------------------------------- schema
class PatientFeatures(BaseModel):
    """Input record. Ranges follow the dataset's data dictionary."""
    model_config = ConfigDict(json_schema_extra={"example": {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0,
        "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}})

    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(..., ge=50, le=300, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: float = Field(..., ge=40, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak-exercise ST segment (0-2)")
    ca: Optional[int] = Field(None, ge=0, le=3, description="Major vessels coloured (0-3); optional")
    thal: Optional[int] = Field(None, ge=1, le=3, description="Thalassemia (1-3); optional")


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    confidence: float
    probability_disease: float
    model_version: str


# ----------------------------------------------------------------- app
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(os.getenv("MODEL_PATH", config.MODEL_PATH))
    state["model"] = load_model(model_path)
    state["meta"] = load_metadata(Path(os.getenv("METADATA_PATH", config.METADATA_PATH)))
    version = state["meta"].get("model_version", __version__)
    MODEL_INFO.labels(state["meta"].get("model_name", "unknown"), version).set(1)
    log_event("model loaded", model_path=str(model_path), model_version=version)
    yield
    state.clear()


app = FastAPI(title="Heart Disease Risk API", version=__version__, lifespan=lifespan,
              description="Predicts the presence of heart disease from 13 clinical features. "
                          "Educational project - not a medical device.")


@app.middleware("http")
async def observe(request: Request, call_next):
    rid = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    request.state.request_id = rid
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = rid
        return response
    finally:
        elapsed = time.perf_counter() - start
        route = request.scope.get("route")
        endpoint = getattr(route, "path", "unmatched")
        REQUESTS.labels(request.method, endpoint, str(status)).inc()
        LATENCY.labels(endpoint).observe(elapsed)
        if endpoint != "/metrics":
            log_event("request", request_id=rid, method=request.method, path=request.url.path,
                      status=status, latency_ms=round(elapsed * 1000, 2),
                      client=request.client.host if request.client else None)


@app.get("/health")
def health():
    return {"status": "ok" if "model" in state else "loading",
            "model_version": state.get("meta", {}).get("model_version", __version__)}


@app.get("/model-info")
def model_info():
    meta = dict(state.get("meta", {}))
    return meta or {"detail": "no metadata available"}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures, request: Request):
    result = predict_records(state["model"], [patient.model_dump()])[0]
    version = state["meta"].get("model_version", __version__)
    PREDICTIONS.labels(result["label"]).inc()
    CONFIDENCE.observe(result["confidence"])
    log_event("prediction", request_id=request.state.request_id,
              prediction=result["prediction"], confidence=result["confidence"], model_version=version)
    return {**result, "model_version": version}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    logger.exception("unhandled error")
    return JSONResponse(status_code=500, content={"detail": "internal server error"})
