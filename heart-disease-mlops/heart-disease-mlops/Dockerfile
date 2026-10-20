# ---- Heart Disease Risk API (inference image) ----
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    MODEL_PATH=/app/models/heart_pipeline.joblib \
    METADATA_PATH=/app/models/metadata.json \
    API_LOG_FILE=/app/logs/api.log

WORKDIR /app

# 1) dependencies first so the layer is cached between code changes
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# 2) application code + trained model artefacts
COPY src/ ./src/
COPY models/heart_pipeline.joblib models/metadata.json ./models/

# 3) run as an unprivileged user
RUN useradd --create-home --uid 10001 appuser && mkdir -p /app/logs && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status==200 else 1)"

CMD ["uvicorn", "heart.api:app", "--host", "0.0.0.0", "--port", "8000"]
