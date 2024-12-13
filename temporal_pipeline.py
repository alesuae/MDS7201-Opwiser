import mlflow
import optuna
import warnings
import pandas as pd
import os

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from src.models.arima_model import SARIMAXModel

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
    X_train, X_test, y_train, y_test, X_train_temp, X_test_temp, y_train_temp, y_test_temp = load_data(splits_path)
    log_data_to_mlflow(pd.DataFrame(X_train), "X_train.csv", "data/splits")
    log_data_to_mlflow(pd.DataFrame(X_test), "X_test.csv", "data/splits")
    log_data_to_mlflow(pd.DataFrame(y_train), "y_train.csv", "data/splits")
    log_data_to_mlflow(pd.DataFrame(y_test), "y_test.csv", "data/splits")

    # Registrar los datos temporales en MLflow
    log_data_to_mlflow(X_train_temp.reset_index(), "X_train_temp.csv", "data/splits")
    log_data_to_mlflow(X_test_temp.reset_index(), "X_test_temp.csv", "data/splits")
    log_data_to_mlflow(y_train_temp.reset_index(), "y_train_temp.csv", "data/splits")  
    log_data_to_mlflow(y_test_temp.reset_index(), "y_test_temp.csv", "data/splits")    

    # 4. Entrenar baseline
    print("Entrenando modelo baseline...")
    params_base = {}
    model_base = DummyRegressor(strategy=config_dict['baseline_strategy'])
    log_model_with_mlflow(
        model = model_base,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        run_name="Baseline",
        params=params_base,
    )

    # AÑADIR ENTRENAMIENTO DE ARIMA
    # Definir columnas exógenas
    exog_columns = [
        "imacec_general", "imacec_comercio", "imacec_no_minero", "ee_comercio",
        "imce_general", "imce_comercio", "icc", "ine_alimentos", "ine_supermercados",
        "tpm", "pib", "es_festivo", "tavg", "tmin", "tmax", 
        "cyber_monday", "black_friday"
    ]

    # Entrenar ARIMA sin variables exógenas
    print("Entrenando modelo ARIMA sin variables exógenas...")
    with mlflow.start_run(run_name="ARIMA_sin_exog"):
        sarimax_no_exog = SARIMAXModel(order=(0, 1, 1), seasonal_order=(0, 1, 1, 52))
        sarimax_no_exog.fit(y_train_temp)
        
        forecast_no_exog = sarimax_no_exog.predict(steps=len(y_test_temp))
        mae_no_exog, rmse_no_exog = sarimax_no_exog.evaluate(y_test_temp, forecast_no_exog)
        
        # Log de métricas y modelo
        mlflow.log_metric("mae", mae_no_exog)
        mlflow.log_metric("rmse", rmse_no_exog)
        mlflow.log_param("model_type", "SARIMAX")
        mlflow.log_param("order", sarimax_no_exog.order)
        mlflow.log_param("seasonal_order", sarimax_no_exog.seasonal_order)

        # Guardar artefactos
        output_path = "src/results/arima_model"
        sarimax_no_exog.save_model(os.path.join(output_path, "sarimax_no_exog.pkl"))
        sarimax_no_exog.save_plot(
            y_train=y_train_temp,
            forecast=forecast_no_exog,
            steps=52,
            file_path=os.path.join(output_path, "sarimax_no_exog_forecast.png")
        )

    # Entrenar ARIMA con variables exógenas
    print("Entrenando modelo ARIMA con variables exógenas...")
    with mlflow.start_run(run_name="ARIMA_con_exog"):
        sarimax_with_exog = SARIMAXModel(order=(0, 1, 1), seasonal_order=(0, 1, 1, 52))
        
        # Asegurar que las columnas exógenas existan en X_train_temp
        exog_train = X_train_temp[exog_columns]
        exog_test = X_test_temp[exog_columns]
        
        sarimax_with_exog.fit(y_train_temp, exog=exog_train)
        
        forecast_with_exog = sarimax_with_exog.predict(steps=len(y_test_temp), exog=exog_test)
        mae_with_exog, rmse_with_exog = sarimax_with_exog.evaluate(y_test_temp, forecast_with_exog)
        
        # Log de métricas y modelo
        mlflow.log_metric("mae", mae_with_exog)
        mlflow.log_metric("rmse", rmse_with_exog)
        mlflow.log_param("model_type", "SARIMAX")
        mlflow.log_param("order", sarimax_with_exog.order)
        mlflow.log_param("seasonal_order", sarimax_with_exog.seasonal_order)
        mlflow.log_param("exog_columns", exog_columns)

        # Guardar artefactos
        sarimax_with_exog.save_model(os.path.join(output_path, "sarimax_with_exog.pkl"))
        sarimax_with_exog.save_plot(
            y_train=y_train_temp,
            forecast=forecast_with_exog,
            steps=52,
            file_path=os.path.join(output_path, "sarimax_with_exog_forecast.png")
        )

    # 5. Modelos de Machine Learning
    print("Entrenando modelos de Machine Learning...")
    seed = 42
    params_ml = {}
    models_to_evaluate = {
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=seed),
        "RandomForestRegressor": RandomForestRegressor(random_state=seed),
        "XGBRegressor": XGBRegressor(random_state=seed),
        "LGBMRegressor": LGBMRegressor(random_state=seed),
    }

    # Iterar sobre los modelos para evaluar los pipelines
    for model_name, model in models_to_evaluate.items():
        log_model_with_mlflow(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            run_name=model_name,
            params=params_ml,
        )

    print("Cargando el mejor modelo ML entrenado...")
    best_model, best_run_id = load_best_model_by_metric(
        experiment_name=experiment_name,
        metric_name=metric_name,
        maximize=False
        )
    print(f"El mejor modelo es: {type(best_model).__name__}")
    print(f"Mejor modelo cargado desde el run: {best_run_id}")
    
    # 6. Optimizar modelo con Optuna
    print("Optimizando modelo con Optuna...")
    param_distributions = get_param_distributions(best_model)

    study_op = optimize_model_with_optuna(
        model_class=best_model,
        param_distributions=param_distributions,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_trials=n_trials,
    )

    # 7. Entrenar el mejor modelo con los mejores hiperparámetros
    print("Entrenando el mejor modelo con los mejores hiperparámetros...")
    best_params = study_op.best_trial.params
    model_optimized = best_model.set_params(**best_params)
    log_model_with_mlflow(
        model=model_optimized,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        run_name=f"Optimized Model {type(best_model).__name__}",
        params=best_params,
        register_model=True
    )

    print("Cargando el mejor modelo entrenado...")
    best_model, best_run_id = load_best_model_by_metric(
        experiment_name=experiment_name,
        metric_name=metric_name,
        maximize=False
        )
    
    print(f"El mejor modelo es: {type(best_model).__name__}")
    print(f"Mejor modelo cargado desde el run: {best_run_id}")

    # 8. Interpretar el modelo con SHAP
    print("Interpretando el modelo con SHAP...")
    X_s = df.drop(columns=["venta_total_neto"])
    log_shap_interpretation(best_model, X_s, num_samples=3)
    print("Pipeline completado.")

if __name__ == "__main__":
    main()




