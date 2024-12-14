import mlflow
import optuna
import warnings
import pandas as pd
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import os
import json

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from src.models.prophet_model import PROPHETModel

from src.mlflow_tracking.tracking import configure_mlflow
from src.mlflow_tracking.model_logger import log_model_with_mlflow
from src.mlflow_tracking.model_loader import load_best_model_by_metric
from src.mlflow_tracking.artifact_logger import log_data_to_mlflow, log_artifact_to_mlflow
from src.mlflow_tracking.optuna_logger import optimize_model_with_optuna
from src.utils.check_data import check_or_create_processed_data, check_or_create_splitted_data
from src.utils.params import get_param_distributions
from src.mlflow_tracking.interpretability import log_shap_interpretation

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np

from src.utils.config import get_config
from src.utils.load_data import load_data

config_dict = get_config('model')
raw_path = config_dict["data"]['raw_data_dir']
processed_path = f"{config_dict['data']['processed_data_dir']}cleaned_data.csv"
splits_path = config_dict["data"]['splits_data_dir']

experiment_name = config_dict['experiment_name']
n_trials = config_dict['n_trials']

metric_name = config_dict['metric_name']

def main():
    # 1. Configurar MLFlow
    configure_mlflow(experiment_name)

    # 2. Verificar si hay datos guardados
    print("Verificando existencia de datos preprocesados...")
    check_or_create_processed_data(raw_path, processed_path)
    df = pd.read_csv(processed_path)
    check_or_create_splitted_data(df, splits_path)

    # 3. Cargar datos
    print("Cargando datos...")
    X_train, X_test, y_train, y_test = load_data(splits_path)
    log_data_to_mlflow(pd.DataFrame(X_train), "X_train.csv", "data/splits")
    log_data_to_mlflow(pd.DataFrame(X_test), "X_test.csv", "data/splits")
    log_data_to_mlflow(pd.DataFrame(y_train), "y_train.csv", "data/splits")
    log_data_to_mlflow(pd.DataFrame(y_test), "y_test.csv", "data/splits")


    # --------------------------------------


    # 1. Modelo Prophet básico (sin regresores exógenos)
    print("Entrenando el modelo Prophet básico...")
    prophet_model_base = PROPHETModel()

    # Medir tiempo de entrenamiento
    start_time = time.time()

    # Entrenar el modelo
    regressors = [col for col in X_train.columns if col.startswith('categoria2_')] + ['stock_disponible_total']
    prophet_model_base.fit(X_train, y_train, regressors)

    # Realizar predicciones y agrupar por semana y regresores
    forecast_train = prophet_model_base.predict(X_train, regressors)
    forecast_train_gr, y_train = prophet_model_base.group_forecast_and_y(forecast_train, y_train, regressors)

    forecast_test = prophet_model_base.predict(X_test, regressors)
    forecast_test_gr, y_test = prophet_model_base.group_forecast_and_y(forecast_test, y_test, regressors)

    # Calcular tiempo total
    execution_time = time.time() - start_time

    # Funciones reales
    y_train_actual = y_train.values
    y_test_actual = y_test.values

    # Predicciones
    y_train_pred = forecast_train_gr['yhat'].values
    y_test_pred = forecast_test_gr['yhat'].values

    # Entrenamiento
    mae_train = mean_absolute_error(y_train_actual, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
    r2_train = r2_score(y_train_actual, y_train_pred)
    wape_train = (np.sum(np.abs(y_train_actual - y_train_pred)) / np.sum(y_train_actual)) * 100

    # Test
    mae_test = mean_absolute_error(y_test_actual, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
    r2_test = r2_score(y_test_actual, y_test_pred)
    wape_test = (np.sum(np.abs(y_test_actual - y_test_pred)) / np.sum(y_test_actual)) * 100

    metrics = {
        "train": {
            "mae": mae_train,
            "rmse": rmse_train,
            "r2": r2_train,
            "wape": wape_train
        },
        "test": {
            "mae": mae_test,
            "rmse": rmse_test,
            "r2": r2_test,
            "wape": wape_test
        },
        "execution_time_seconds": execution_time
    }

    # Guardar métricas en un archivo JSON
    output_path = os.path.join(os.path.dirname(__file__), "prophet_base_metrics.json")
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Métricas guardadas en {output_path}")


    # --------------------------------------

    # Variables exógenas
    variables = {
        "macroeconomicas": [
            "icc", 
            "imacec_comercio", 
            "imacec_general", 
            "imacec_no_minero", 
            "imce_comercio", 
            "imce_general", 
            "ine_alimentos", 
            "ine_supermercados", 
            "pib"
        ],
        "climaticas": [
            "tavg", 
            "tmax", 
            "tmin"
        ],
        "dias_especiales": [
            "black_friday", 
            "cyber_monday", 
            "es_festivo"
        ]
    }

    # --------------------------------------

    # Cargar datos
    print("Cargando datos...")
    X_train, X_test, y_train, y_test = load_data(splits_path)
    # 2. Modelo Prophet con variables macroeconómicas
    print("Entrenando el modelo Prophet con variables macroeconómicas...")
    prophet_model_macro = PROPHETModel()

    # Medir tiempo de entrenamiento
    start_time = time.time()

    # Entrenar el modelo
    regressors = [col for col in X_train.columns if col.startswith('categoria2_')] + ['stock_disponible_total'] + variables['macroeconomicas']
    prophet_model_macro.fit(X_train, y_train, regressors)

    # Realizar predicciones y agrupar por semana y regresores
    forecast_train = prophet_model_macro.predict(X_train, regressors)
    forecast_train_gr, y_train = prophet_model_macro.group_forecast_and_y(forecast_train, y_train, regressors)

    forecast_test = prophet_model_macro.predict(X_test, regressors)
    forecast_test_gr, y_test = prophet_model_macro.group_forecast_and_y(forecast_test, y_test, regressors)

    # Calcular tiempo total
    execution_time = time.time() - start_time

    # Funciones reales
    y_train_actual = y_train.values
    y_test_actual = y_test.values

    # Predicciones
    y_train_pred = forecast_train_gr['yhat'].values
    y_test_pred = forecast_test_gr['yhat'].values

    # Entrenamiento
    mae_train = mean_absolute_error(y_train_actual, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
    r2_train = r2_score(y_train_actual, y_train_pred)
    wape_train = (np.sum(np.abs(y_train_actual - y_train_pred)) / np.sum(y_train_actual)) * 100

    # Test
    mae_test = mean_absolute_error(y_test_actual, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
    r2_test = r2_score(y_test_actual, y_test_pred)
    wape_test = (np.sum(np.abs(y_test_actual - y_test_pred)) / np.sum(y_test_actual)) * 100

    metrics = {
        "train": {
            "mae": mae_train,
            "rmse": rmse_train,
            "r2": r2_train,
            "wape": wape_train
        },
        "test": {
            "mae": mae_test,
            "rmse": rmse_test,
            "r2": r2_test,
            "wape": wape_test
        },
        "execution_time_seconds": execution_time
    }

    # Guardar métricas en un archivo JSON
    output_path = os.path.join(os.path.dirname(__file__), "prophet_macro_metrics.json")
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Métricas guardadas en {output_path}")

    



    print("Cargando datos...")
    X_train, X_test, y_train, y_test = load_data(splits_path)
    # 3. Modelo Prophet con variables climáticas
    print("Entrenando el modelo Prophet con variables climáticas...")
    prophet_model_clima = PROPHETModel()

    # Medir tiempo de entrenamiento
    start_time = time.time()

    # Entrenar el modelo
    regressors = [col for col in X_train.columns if col.startswith('categoria2_')] + ['stock_disponible_total'] + variables['climaticas']
    prophet_model_clima.fit(X_train, y_train, regressors)

    # Realizar predicciones y agrupar por semana y regresores
    forecast_train = prophet_model_clima.predict(X_train, regressors)
    forecast_train_gr, y_train = prophet_model_clima.group_forecast_and_y(forecast_train, y_train, regressors)

    forecast_test = prophet_model_clima.predict(X_test, regressors)
    forecast_test_gr, y_test = prophet_model_clima.group_forecast_and_y(forecast_test, y_test, regressors)

    # Calcular tiempo total
    execution_time = time.time() - start_time

    # Funciones reales
    y_train_actual = y_train.values
    y_test_actual = y_test.values

    # Predicciones
    y_train_pred = forecast_train_gr['yhat'].values
    y_test_pred = forecast_test_gr['yhat'].values

    # Entrenamiento
    mae_train = mean_absolute_error(y_train_actual, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
    r2_train = r2_score(y_train_actual, y_train_pred)
    wape_train = (np.sum(np.abs(y_train_actual - y_train_pred)) / np.sum(y_train_actual)) * 100

    # Test
    mae_test = mean_absolute_error(y_test_actual, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
    r2_test = r2_score(y_test_actual, y_test_pred)
    wape_test = (np.sum(np.abs(y_test_actual - y_test_pred)) / np.sum(y_test_actual)) * 100

    metrics = {
        "train": {
            "mae": mae_train,
            "rmse": rmse_train,
            "r2": r2_train,
            "wape": wape_train
        },
        "test": {
            "mae": mae_test,
            "rmse": rmse_test,
            "r2": r2_test,
            "wape": wape_test
        },
        "execution_time_seconds": execution_time
    }

    # Guardar métricas en un archivo JSON
    output_path = os.path.join(os.path.dirname(__file__), "prophet_clima_metrics.json")
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Métricas guardadas en {output_path}")





    print("Cargando datos...")
    X_train, X_test, y_train, y_test = load_data(splits_path)
    # 4. Modelo Prophet con variables de días especiales
    print("Entrenando el modelo Prophet con variables de días especiales...")
    prophet_model_dias = PROPHETModel()

    # Medir tiempo de entrenamiento
    start_time = time.time()

    # Entrenar el modelo
    regressors = [col for col in X_train.columns if col.startswith('categoria2_')] + ['stock_disponible_total'] + variables['dias_especiales']
    prophet_model_dias.fit(X_train, y_train, regressors)

    # Realizar predicciones y agrupar por semana y regresores
    forecast_train = prophet_model_dias.predict(X_train, regressors)
    forecast_train_gr, y_train = prophet_model_dias.group_forecast_and_y(forecast_train, y_train, regressors)

    forecast_test = prophet_model_dias.predict(X_test, regressors)
    forecast_test_gr, y_test = prophet_model_dias.group_forecast_and_y(forecast_test, y_test, regressors)

    # Calcular tiempo total
    execution_time = time.time() - start_time

    # Funciones reales
    y_train_actual = y_train.values
    y_test_actual = y_test.values

    # Predicciones
    y_train_pred = forecast_train_gr['yhat'].values
    y_test_pred = forecast_test_gr['yhat'].values

    # Entrenamiento
    mae_train = mean_absolute_error(y_train_actual, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
    r2_train = r2_score(y_train_actual, y_train_pred)
    wape_train = (np.sum(np.abs(y_train_actual - y_train_pred)) / np.sum(y_train_actual)) * 100

    # Test
    mae_test = mean_absolute_error(y_test_actual, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
    r2_test = r2_score(y_test_actual, y_test_pred)
    wape_test = (np.sum(np.abs(y_test_actual - y_test_pred)) / np.sum(y_test_actual)) * 100

    metrics = {
        "train": {
            "mae": mae_train,
            "rmse": rmse_train,
            "r2": r2_train,
            "wape": wape_train
        },
        "test": {
            "mae": mae_test,
            "rmse": rmse_test,
            "r2": r2_test,
            "wape": wape_test
        },
        "execution_time_seconds": execution_time
    }

    # Guardar métricas en un archivo JSON
    output_path = os.path.join(os.path.dirname(__file__), "prophet_dias_metrics.json")
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Métricas guardadas en {output_path}")





    print("Cargando datos...")
    X_train, X_test, y_train, y_test = load_data(splits_path)
    # 5. Modelo Prophet con todas las variables exógenas
    print("Entrenando el modelo Prophet con todas las variables exógenas...")
    prophet_model_all = PROPHETModel()

    # Medir tiempo de entrenamiento
    start_time = time.time()

    # Entrenar el modelo
    regressors = [col for col in X_train.columns if col.startswith('categoria2_')] + ['stock_disponible_total'] + variables['macroeconomicas'] + variables['climaticas'] + variables['dias_especiales']
    prophet_model_all.fit(X_train, y_train, regressors)

    # Realizar predicciones y agrupar por semana y regresores
    forecast_train = prophet_model_all.predict(X_train, regressors)
    forecast_train_gr, y_train = prophet_model_all.group_forecast_and_y(forecast_train, y_train, regressors)

    forecast_test = prophet_model_all.predict(X_test, regressors)
    forecast_test_gr, y_test = prophet_model_all.group_forecast_and_y(forecast_test, y_test, regressors)

    # Calcular tiempo total
    execution_time = time.time() - start_time

    # Funciones reales
    y_train_actual = y_train.values
    y_test_actual = y_test.values

    # Predicciones
    y_train_pred = forecast_train_gr['yhat'].values
    y_test_pred = forecast_test_gr['yhat'].values

    # Entrenamiento
    mae_train = mean_absolute_error(y_train_actual, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
    r2_train = r2_score(y_train_actual, y_train_pred)
    wape_train = (np.sum(np.abs(y_train_actual - y_train_pred)) / np.sum(y_train_actual)) * 100

    # Test
    mae_test = mean_absolute_error(y_test_actual, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
    r2_test = r2_score(y_test_actual, y_test_pred)
    wape_test = (np.sum(np.abs(y_test_actual - y_test_pred)) / np.sum(y_test_actual)) * 100

    metrics = {
        "train": {
            "mae": mae_train,
            "rmse": rmse_train,
            "r2": r2_train,
            "wape": wape_train
        },
        "test": {
            "mae": mae_test,
            "rmse": rmse_test,
            "r2": r2_test,
            "wape": wape_test
        },
        "execution_time_seconds": execution_time
    }

    # Guardar métricas en un archivo JSON
    output_path = os.path.join(os.path.dirname(__file__), "prophet_all_metrics.json")
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Métricas guardadas en {output_path}")


    # Las métricas fueron dejadas en el directorio "src\results\prophet_model\metrics"


    return

if __name__ == "__main__":
    main()




