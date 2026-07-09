.PHONY: help install lint test train deploy inference validate app clean

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install project dependencies
	python -m pip install --upgrade pip
	pip install -r requirements.txt

lint:  ## Run flake8 across the project
	flake8 steps model pipelines tests *.py

test:  ## Run the test suite
	python -m pytest --import-mode=append tests/

train:  ## Run the training pipeline
	python run_training_pipeline.py

deploy:  ## Run the continuous deployment pipeline
	python run_deployment_pipeline.py

inference:  ## Run the inference pipeline
	python run_inference_pipeline.py

validate:  ## Run deepchecks data validation
	python run_data_validation.py

app:  ## Launch the Streamlit app
	streamlit run streamlit_app.py

clean:  ## Remove caches and build artifacts
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	rm -rf .pytest_cache
