import sys
import os
import json
# import matplotlib.pyplot as plt
# import pandas as pd

# Configurar directorios de trabajo
current_dir = os.path.dirname(__file__)
src_path = os.path.abspath(os.path.join(current_dir, "../.."))
data_path = os.path.abspath(os.path.join(current_dir, "../../data"))

sys.path.append(src_path)
sys.path.append(data_path)

# logs_path = os.path.join(current_dir, "logs")
metrics_path = os.path.join(current_dir, "metrics")
models_path = os.path.join(current_dir, "models")
plots_path = os.path.join(current_dir, "plots")

# os.makedirs(logs_path, exist_ok=True)
os.makedirs(metrics_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)
os.makedirs(plots_path, exist_ok=True)

from data.main import data_pipeline
from models.prophet_model import PROPHETModel

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Ejecutar el pipeline para cargar los conjuntos de datos
print("Ejecutando el pipeline para cargar los datos...")
X_train, X_val, X_test, y_train, y_val, y_test = data_pipeline()

# 1. Modelo Prophet básico (sin regresores exógenos)
print("Entrenando el modelo Prophet básico...")
prophet_model_base = PROPHETModel()

# Entrenar el modelo
regressors = [col for col in X_train.columns if col.startswith('linea2_')] + ['stock_disponible_total']
prophet_model_base.fit(X_train, y_train, regressors)

# Realizar predicciones y agrupar por semana y regresores
forecast_train = prophet_model_base.predict(X_train, regressors)
forecast_train_gr, y_train = prophet_model_base.group_forecast_and_y(forecast_train, y_train, regressors)

forecast_val = prophet_model_base.predict(X_val, regressors)
forecast_val_gr, y_val = prophet_model_base.group_forecast_and_y(forecast_val, y_val, regressors)

forecast_test = prophet_model_base.predict(X_test, regressors)
forecast_test_gr, y_test = prophet_model_base.group_forecast_and_y(forecast_test, y_test, regressors)

# Calcular errores usando el método de la clase PROPHETModel
mae_train, rmse_train = prophet_model_base.calculate_mae_rmse(y_train, forecast_train_gr['yhat'])
mae_val, rmse_val = prophet_model_base.calculate_mae_rmse(y_val, forecast_val_gr['yhat'])
mae_test, rmse_test = prophet_model_base.calculate_mae_rmse(y_test, forecast_test_gr['yhat'])

# Guardar métricas en JSON
metrics = {
    "train": {"mae": mae_train, "rmse": rmse_train},
    "val": {"mae": mae_val, "rmse": rmse_val},
    "test": {"mae": mae_test, "rmse": rmse_test}
}

with open(os.path.join(metrics_path, "prophet_base_metrics.json"), 'w') as f:
    json.dump(metrics, f)

# Guardar el modelo entrenado
prophet_model_base.save_model(os.path.join(models_path, "prophet_base_model.pkl"))

# Graficar las predicciones y componentes
prophet_model_base.plot(
    forecast_train, 
    title="Predicciones en Conjunto de Entrenamiento (Base)",
    save_path=os.path.join(plots_path, "prophet_base_train_predictions.png")
)
prophet_model_base.plot_components(
    forecast_train, 
    title="Componentes en Conjunto de Entrenamiento (Base)",
    save_path=os.path.join(plots_path, "prophet_base_train_components.png")
)
prophet_model_base.plot(
    forecast_val, 
    title="Predicciones en Conjunto de Validación (Base)",
    save_path=os.path.join(plots_path, "prophet_base_val_predictions.png")
)
prophet_model_base.plot_components(
    forecast_val, 
    title="Componentes en Conjunto de Validación (Base)",
    save_path=os.path.join(plots_path, "prophet_base_val_components.png")
)
prophet_model_base.plot(
    forecast_test, 
    title="Predicciones en Conjunto de Prueba (Base)",
    save_path=os.path.join(plots_path, "prophet_base_test_predictions.png")
)
prophet_model_base.plot_components(
    forecast_test, 
    title="Componentes en Conjunto de Prueba (Base)",
    save_path=os.path.join(plots_path, "prophet_base_test_components.png")
)

# # 2. Modelo Prophet con optimización de hiperparámetros
# print("Entrenando el modelo Prophet con optimización de hiperparámetros...")

# # Definir la grilla de parámetros
# param_grid = {
#     'seasonality_mode': ['additive', 'multiplicative'],
#     'changepoint_prior_scale': [0.01, 0.1, 0.5],
#     'yearly_seasonality': [True, False],
#     'weekly_seasonality': [True, False],
#     'daily_seasonality': [True, False]
# }

# # Llamar a la función de optimización
# tuning_results, best_params = prophet_model_base.optimize_hyperparameters(X_train, X_val, y_train, y_val, param_grid, features=[])

# # Guardar los resultados de la optimización
# tuning_results.to_csv(os.path.join(logs_path, "prophet_tuning_results.csv"))
# with open(os.path.join(logs_path, "best_params.json"), 'w') as f:
#     json.dump(best_params, f)

# # Crear el modelo Prophet con los mejores parámetros
# prophet_model_optimized = PROPHETModel(**best_params)
# prophet_model_optimized.fit(X_train, y_train)

# # Realizar predicciones
# forecast_train_opt = prophet_model_optimized.predict(X_train)
# forecast_val_opt = prophet_model_optimized.predict(X_val)
# forecast_test_opt = prophet_model_optimized.predict(X_test)

# # Calcular errores usando el método de la clase PROPHETModel
# mae_train_opt, rmse_train_opt = prophet_model_optimized.calculate_mae_rmse(y_train, forecast_train_opt['yhat'])
# mae_val_opt, rmse_val_opt = prophet_model_optimized.calculate_mae_rmse(y_val, forecast_val_opt['yhat'])
# mae_test_opt, rmse_test_opt = prophet_model_optimized.calculate_mae_rmse(y_test, forecast_test_opt['yhat'])

# # Guardar métricas optimizadas en JSON
# metrics_optimized = {
#     "train": {"mae": mae_train_opt, "rmse": rmse_train_opt},
#     "val": {"mae": mae_val_opt, "rmse": rmse_val_opt},
#     "test": {"mae": mae_test_opt, "rmse": rmse_test_opt}
# }

# with open(os.path.join(metrics_path, "prophet_optimized_metrics.json"), 'w') as f:
#     json.dump(metrics_optimized, f)

# # Guardar el modelo optimizado
# prophet_model_optimized.save_model(os.path.join(models_path, "prophet_optimized_model.pkl"))

# # Graficar las predicciones y componentes del modelo optimizado
# prophet_model_optimized.plot(
#     forecast_train_opt,
#     title="Predicciones en Conjunto de Entrenamiento (Optimizado)",
#     save_path=os.path.join(plots_path, "prophet_optimized_train_predictions.png")
# )
# prophet_model_optimized.plot_components(
#     forecast_train_opt,
#     title="Componentes en Conjunto de Entrenamiento (Optimizado)",
#     save_path=os.path.join(plots_path, "prophet_optimized_train_components.png")
# )
# prophet_model_optimized.plot(
#     forecast_val_opt,
#     title="Predicciones en Conjunto de Validación (Optimizado)",
#     save_path=os.path.join(plots_path, "prophet_optimized_val_predictions.png")
# )
# prophet_model_optimized.plot_components(
#     forecast_val_opt,
#     title="Componentes en Conjunto de Validación (Optimizado)",
#     save_path=os.path.join(plots_path, "prophet_optimized_val_components.png")
# )
# prophet_model_optimized.plot(
#     forecast_test_opt,
#     title="Predicciones en Conjunto de Prueba (Optimizado)",
#     save_path=os.path.join(plots_path, "prophet_optimized_test_predictions.png")
# )
# prophet_model_optimized.plot_components(
#     forecast_test_opt,
#     title="Componentes en Conjunto de Prueba (Optimizado)",
#     save_path=os.path.join(plots_path, "prophet_optimized_test_components.png")
# )

# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------

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

# Ejecutar el pipeline para cargar los conjuntos de datos
print("Ejecutando el pipeline para cargar los datos...")
X_train, X_val, X_test, y_train, y_val, y_test = data_pipeline()

# 2. Modelo Prophet con variables macroeconómicas
print("Entrenando el modelo Prophet con variables macroeconómicas...")
prophet_model_macro = PROPHETModel()

# Entrenar el modelo
regressors = [col for col in X_train.columns if col.startswith('linea2_')] + ['stock_disponible_total'] + variables['macroeconomicas']
prophet_model_macro.fit(X_train, y_train, regressors)

# Realizar predicciones y agrupar por semana y regresores
forecast_train = prophet_model_macro.predict(X_train, regressors)
forecast_train, y_train = prophet_model_macro.group_forecast_and_y(forecast_train, y_train, regressors)

forecast_val = prophet_model_macro.predict(X_val, regressors)
forecast_val, y_val = prophet_model_macro.group_forecast_and_y(forecast_val, y_val, regressors)

forecast_test = prophet_model_macro.predict(X_test, regressors)
forecast_test, y_test = prophet_model_macro.group_forecast_and_y(forecast_test, y_test, regressors)

# Calcular errores usando el método de la clase PROPHETModel
mae_train, rmse_train = prophet_model_macro.calculate_mae_rmse(y_train, forecast_train['yhat'])
mae_val, rmse_val = prophet_model_macro.calculate_mae_rmse(y_val, forecast_val['yhat'])
mae_test, rmse_test = prophet_model_macro.calculate_mae_rmse(y_test, forecast_test['yhat'])

# Guardar métricas en JSON
metrics = {
    "train": {"mae": mae_train, "rmse": rmse_train},
    "val": {"mae": mae_val, "rmse": rmse_val},
    "test": {"mae": mae_test, "rmse": rmse_test}
}

with open(os.path.join(metrics_path, "prophet_macro_metrics.json"), 'w') as f:
    json.dump(metrics, f)

# Guardar el modelo entrenado
prophet_model_macro.save_model(os.path.join(models_path, "prophet_macro_model.pkl"))

# -----------------------------------------------------------------------------------------------------------------

# Ejecutar el pipeline para cargar los conjuntos de datos
print("Ejecutando el pipeline para cargar los datos...")
X_train, X_val, X_test, y_train, y_val, y_test = data_pipeline()

# 3. Modelo Prophet con variables climáticas
print("Entrenando el modelo Prophet con variables climáticas...")
prophet_model_clima = PROPHETModel()

# Entrenar el modelo
regressors = [col for col in X_train.columns if col.startswith('linea2_')] + ['stock_disponible_total'] + variables['climaticas']
prophet_model_clima.fit(X_train, y_train, regressors)

# Realizar predicciones y agrupar por semana y regresores
forecast_train = prophet_model_clima.predict(X_train, regressors)
forecast_train_gr, y_train = prophet_model_clima.group_forecast_and_y(forecast_train, y_train, regressors)

forecast_val = prophet_model_clima.predict(X_val, regressors)
forecast_val_gr, y_val = prophet_model_clima.group_forecast_and_y(forecast_val, y_val, regressors)

forecast_test = prophet_model_clima.predict(X_test, regressors)
forecast_test_gr, y_test = prophet_model_clima.group_forecast_and_y(forecast_test, y_test, regressors)

# Calcular errores usando el método de la clase PROPHETModel
mae_train, rmse_train = prophet_model_clima.calculate_mae_rmse(y_train, forecast_train_gr['yhat'])
mae_val, rmse_val = prophet_model_clima.calculate_mae_rmse(y_val, forecast_val_gr['yhat'])
mae_test, rmse_test = prophet_model_clima.calculate_mae_rmse(y_test, forecast_test_gr['yhat'])

# Guardar métricas en JSON
metrics = {
    "train": {"mae": mae_train, "rmse": rmse_train},
    "val": {"mae": mae_val, "rmse": rmse_val},
    "test": {"mae": mae_test, "rmse": rmse_test}
}

with open(os.path.join(metrics_path, "prophet_clima_metrics.json"), 'w') as f:
    json.dump(metrics, f)

# Guardar el modelo entrenado
prophet_model_clima.save_model(os.path.join(models_path, "prophet_clima_model.pkl"))

# Graficar las predicciones y componentes
prophet_model_clima.plot(
    forecast_train, 
    title="Predicciones en Conjunto de Entrenamiento (Clima)",
    save_path=os.path.join(plots_path, "prophet_clima_train_predictions.png")
)
prophet_model_clima.plot_components(
    forecast_train, 
    title="Componentes en Conjunto de Entrenamiento (Clima)",
    save_path=os.path.join(plots_path, "prophet_clima_train_components.png")
)
prophet_model_clima.plot(
    forecast_val, 
    title="Predicciones en Conjunto de Validación (Clima)",
    save_path=os.path.join(plots_path, "prophet_clima_val_predictions.png")
)
prophet_model_clima.plot_components(
    forecast_val, 
    title="Componentes en Conjunto de Validación (Clima)",
    save_path=os.path.join(plots_path, "prophet_clima_val_components.png")
)
prophet_model_clima.plot(
    forecast_test, 
    title="Predicciones en Conjunto de Prueba (Clima)",
    save_path=os.path.join(plots_path, "prophet_clima_test_predictions.png")
)
prophet_model_clima.plot_components(
    forecast_test, 
    title="Componentes en Conjunto de Prueba (Clima)",
    save_path=os.path.join(plots_path, "prophet_clima_test_components.png")
)

# -----------------------------------------------------------------------------------------------------------------

# Ejecutar el pipeline para cargar los conjuntos de datos
print("Ejecutando el pipeline para cargar los datos...")
X_train, X_val, X_test, y_train, y_val, y_test = data_pipeline()

# 4. Modelo Prophet con variables de días especiales
print("Entrenando el modelo Prophet con variables de días especiales...")
prophet_model_dias = PROPHETModel()

# Entrenar el modelo
regressors = [col for col in X_train.columns if col.startswith('linea2_')] + ['stock_disponible_total'] + variables['dias_especiales']
prophet_model_dias.fit(X_train, y_train, regressors)

# Realizar predicciones y agrupar por semana y regresores
forecast_train = prophet_model_dias.predict(X_train, regressors)
forecast_train, y_train = prophet_model_dias.group_forecast_and_y(forecast_train, y_train, regressors)

forecast_val = prophet_model_dias.predict(X_val, regressors)
forecast_val, y_val = prophet_model_dias.group_forecast_and_y(forecast_val, y_val, regressors)

forecast_test = prophet_model_dias.predict(X_test, regressors)
forecast_test, y_test = prophet_model_dias.group_forecast_and_y(forecast_test, y_test, regressors)

# Calcular errores usando el método de la clase PROPHETModel
mae_train, rmse_train = prophet_model_dias.calculate_mae_rmse(y_train, forecast_train['yhat'])
mae_val, rmse_val = prophet_model_dias.calculate_mae_rmse(y_val, forecast_val['yhat'])
mae_test, rmse_test = prophet_model_dias.calculate_mae_rmse(y_test, forecast_test['yhat'])

# Guardar métricas en JSON
metrics = {
    "train": {"mae": mae_train, "rmse": rmse_train},
    "val": {"mae": mae_val, "rmse": rmse_val},
    "test": {"mae": mae_test, "rmse": rmse_test}
}

with open(os.path.join(metrics_path, "prophet_dias_metrics.json"), 'w') as f:
    json.dump(metrics, f)

# Guardar el modelo entrenado
prophet_model_dias.save_model(os.path.join(models_path, "prophet_dias_model.pkl"))

# -----------------------------------------------------------------------------------------------------------------

# Ejecutar el pipeline para cargar los conjuntos de datos
print("Ejecutando el pipeline para cargar los datos...")
X_train, X_val, X_test, y_train, y_val, y_test = data_pipeline()

# 5. Modelo Prophet con todas las variables exógenas
print("Entrenando el modelo Prophet con todas las variables exógenas...")
prophet_model_all = PROPHETModel()

# Entrenar el modelo
regressors = [col for col in X_train.columns if col.startswith('linea2_')] + ['stock_disponible_total'] + variables['macroeconomicas'] + variables['climaticas'] + variables['dias_especiales']
prophet_model_all.fit(X_train, y_train, regressors)

# Realizar predicciones y agrupar por semana y regresores
forecast_train = prophet_model_all.predict(X_train, regressors)
forecast_train, y_train = prophet_model_all.group_forecast_and_y(forecast_train, y_train, regressors)

forecast_val = prophet_model_all.predict(X_val, regressors)
forecast_val, y_val = prophet_model_all.group_forecast_and_y(forecast_val, y_val, regressors)

forecast_test = prophet_model_all.predict(X_test, regressors)
forecast_test, y_test = prophet_model_all.group_forecast_and_y(forecast_test, y_test, regressors)

# Calcular errores usando el método de la clase PROPHETModel
mae_train, rmse_train = prophet_model_all.calculate_mae_rmse(y_train, forecast_train['yhat'])
mae_val, rmse_val = prophet_model_all.calculate_mae_rmse(y_val, forecast_val['yhat'])
mae_test, rmse_test = prophet_model_all.calculate_mae_rmse(y_test, forecast_test['yhat'])

# Guardar métricas en JSON
metrics = {
    "train": {"mae": mae_train, "rmse": rmse_train},
    "val": {"mae": mae_val, "rmse": rmse_val},
    "test": {"mae": mae_test, "rmse": rmse_test}
}

with open(os.path.join(metrics_path, "prophet_all_metrics.json"), 'w') as f:
    json.dump(metrics, f)

# Guardar el modelo entrenado
prophet_model_all.save_model(os.path.join(models_path, "prophet_all_model.pkl"))