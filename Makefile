PYTHON := python
MAIN := main_pipeline.py
REQUIREMENTS := requirements.txt
VENV := .venv
MLFLOW_HOST := 127.0.0.1
MLFLOW_PORT := 5000
MLFLOW_ARTIFACTS := mlruns

help:
	@echo "Usage:"
	@echo "  make install        Install dependencies from requirements.txt"
	@echo "  make run            Run the main pipeline script"
	@echo "  make mlflow         Start the MLflow server"
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

# Limpiar artefactos y registros de MLflow
clean-mlflow:
	@echo "Cleaning MLflow artifacts and database..."
	rm -rf $(MLFLOW_ARTIFACTS)
	rm -f mlflow.db

# Default goal
.DEFAULT_GOAL := help
