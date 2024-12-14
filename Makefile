PYTHON := python
MAIN := main_pipeline.py
REQUIREMENTS := requirements.txt
VENV := .venv
MLFLOW_HOST := 127.0.0.1
MLFLOW_PORT := 5000
MLFLOW_ARTIFACTS := mlruns
PROCESSED_DIR = data/processed
SPLITS_DIR = data/splits
OTHER_DIR = data/other

help:
	@echo "Usage:"
	@echo "  make install        Install dependencies from requirements.txt"
	@echo "  make run            Run the main ML pipeline script"
	@echo "  make run-temporal   Run the temporal pipeline script"
	@echo "  make mlflow         Start the MLflow server"
	@echo "  make auto-eda       Run the auto-EDA script"
	@echo "  make clean          Clean auto-generated files and artifacts"
	@echo "  make clean-mlflow   Clean MLflow artifacts and logs"
	@echo "  make help           Show this help message"

# Instalar dependencias
install:
	@echo "Creating virtual environment and installing dependencies..."
	@if [ ! -d "$(VENV)" ]; then python -m venv $(VENV); fi
	$(VENV)/bin/pip install -r $(REQUIREMENTS)

# Ejecutar el pipeline principal
run:
	@echo "Running the main pipeline..."
	$(PYTHON) $(MAIN)

# Ejecutar el pipeline temporal
run-temporal:
	@echo "Running the temporal pipeline..."
	$(PYTHON) temporal_pipeline.py

auto-eda:
	@echo "Running the automatic EDA report..."
	$(PYTHON) eda_auto.py
	
# Iniciar el servidor de MLflow
mlflow:
	@echo "Starting the MLflow server..."
	mlflow ui

# Limpiar archivos generados automáticamente
clean:
	@echo "Cleaning up generated files..."
	rm -rf __pycache__ *.pyc *.pyo .pytest_cache .venv *.log
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -exec rm -r {} +
	find . -type f -name "*.pyo" -exec rm -r {} +
	@echo "Cleaning processed and split data directories..."
	@if [ -d $(PROCESSED_DIR) ]; then rm -rf $(PROCESSED_DIR)/*; fi
	@if [ -d $(SPLITS_DIR) ]; then rm -rf $(SPLITS_DIR)/*; fi
	@if [ -d $(OTHER_DIR) ]; then rm -rf $(OTHER_DIR)/*; fi
	@echo "Cleanup complete!"

# Limpiar artefactos y registros de MLflow
clean-mlflow:
	@echo "Cleaning MLflow artifacts and database..."
	rm -rf $(MLFLOW_ARTIFACTS)
	rm -f mlflow.db

# Default goal
.DEFAULT_GOAL := help
