.PHONY: install data eda train test lint serve docker-build docker-run k8s-deploy monitoring clean mlflow-ui
export PYTHONPATH := src

install:        ## create venv deps
	pip install -r requirements-dev.txt
data:           ## download + clean dataset
	python scripts/download_data.py && python -m heart.data
eda:
	python -m heart.eda
train:          ## train, tune, track in MLflow, save model
	python -m heart.train
test:
	pytest --cov=src/heart
lint:
	flake8 src tests scripts
serve:          ## run the API locally on :8000
	uvicorn heart.api:app --reload --port 8000
mlflow-ui:      ## browse experiments on :5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
docker-build:
	docker build -t heart-disease-api:1.0.0 .
docker-run:
	docker run --rm -p 8000:8000 heart-disease-api:1.0.0
monitoring:     ## API + Prometheus + Grafana
	docker compose up --build
k8s-deploy:
	kubectl apply -k k8s/
clean:
	rm -rf .pytest_cache .coverage coverage.xml htmlcov
