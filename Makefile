.PHONY: help install data train evaluate monitor leakage serve test lint pipeline \
        synthetic drift-demo clean docker-build docker-run

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Install package + dev dependencies
	pip install -e ".[dev]"

data:  ## Prepare the active dataset (real Telco extract, or the generator)
	churn generate

train:  ## Train the model and log to MLflow
	python -m churn.train

evaluate:  ## Evaluate the saved model
	churn evaluate

monitor:  ## Run the data-drift report
	churn monitor

leakage:  ## Show why the vendor's outcome-derived columns are excluded
	python scripts/leakage_experiment.py

drift-demo:  ## Rebuild 'current' as a skewed cohort, then detect the drift
	python scripts/prepare_telco.py --drift-scenario
	churn monitor

synthetic:  ## Run the whole pipeline on generated data instead of the real extract
	CHURN_DATA_SOURCE=synthetic python scripts/generate_data.py
	CHURN_DATA_SOURCE=synthetic python -m churn.train
	CHURN_DATA_SOURCE=synthetic churn monitor

serve:  ## Run the FastAPI model server
	uvicorn churn.api.main:app --host 0.0.0.0 --port 8000 --reload

test:  ## Run the test suite
	python -m pytest --cov=churn --cov-report=term-missing

lint:  ## Lint with ruff
	python -m ruff check src tests scripts

pipeline: data train evaluate monitor  ## Run the full offline pipeline end to end

clean:  ## Remove generated artifacts (keeps the raw extract)
	rm -rf mlruns mlflow.db models reports .pytest_cache .ruff_cache htmlcov .coverage
	rm -f data/*.csv

docker-build:  ## Build the serving image
	docker build -t churn-mlops:latest .

docker-run:  ## Run the serving image
	docker run --rm -p 8000:8000 -v $(PWD)/models:/app/models:ro churn-mlops:latest
