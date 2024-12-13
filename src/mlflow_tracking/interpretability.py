import mlflow
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import is_regressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def log_shap_interpretation(model, dataset, num_samples=3):
    """
    Genera y registra explicaciones SHAP para un modelo de regresión registrado en MLFlow.

    Args:
        model_name (str): Nombre del modelo registrado en MLFlow.
        dataset (pd.DataFrame): Conjunto de datos de entrada para las explicaciones.
        num_samples (int): Número de instancias para las cuales se calcularán los SHAP values.
    """
    # Cargar el modelo desde MLFlow
    #model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")

    # Seleccionar instancias aleatorias del dataset
    random_indices = np.random.choice(dataset.index, size=num_samples, replace=False)
    sample_data = dataset

    # Inicializar el explainer SHAP basado en el tipo de modelo
    explainer = None
    if isinstance(model, (RandomForestRegressor, XGBRegressor, LGBMRegressor)):
        explainer = shap.TreeExplainer(model)
    elif is_regressor(model):
        explainer = shap.KernelExplainer(model.predict, dataset)
    else:
        raise ValueError("El modelo cargado no es compatible con SHAP interpretability.")

    # Generar valores de SHAP para las instancias seleccionadas
    shap_values = explainer.shap_values(sample_data)
    print(len(shap_values))
    print(len(dataset))

    # Crear gráficos de SHAP
    print("Generando gráficos de SHAP...")
    
    # Resumen de todas las características
    shap.summary_plot(shap_values, dataset, show=False)
    summary_plot_path = "shap_summary_plot.png"
    plt.tight_layout()
    plt.savefig(summary_plot_path)
    plt.close()
    print(f"Gráfico de resumen SHAP guardado en: {summary_plot_path}")

    # Gráficos SHAP para instancias seleccionadas
    for i, idx in enumerate(random_indices):
        shap.force_plot(
            explainer.expected_value,
            shap_values[i],
            sample_data.iloc[i],
            matplotlib=True,
            show=False
        )
        plt.gcf().set_size_inches(20, 8) 
        force_plot_path = f"shap_force_plot_{i}.png"
        plt.savefig(force_plot_path, dpi=300)
        plt.close()
        print(f"Gráfico SHAP de instancia {i + 1} guardado en: {force_plot_path}")

        # Registrar cada gráfico en MLFlow
        with mlflow.start_run(run_name="SHAP Interpretability"):
            mlflow.log_artifact(summary_plot_path, artifact_path="shap_plots")
            mlflow.log_artifact(force_plot_path, artifact_path=f"shap_plots/instance_{i + 1}")
            #mlflow.log_artifact(waterfall_plot_path, artifact_path=f"shap_plots/instance_{i + 1}")

    print("Gráficos de SHAP registrados en MLFlow.")

