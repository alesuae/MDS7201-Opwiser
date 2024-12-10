from src.models.arima_model import SARIMAXModel
from src.data.main import data_pipeline
import os

# Crear las carpetas si no existen
BASE_DIR = "src/results/arima_model"
LOGS_DIR = os.path.join(BASE_DIR, "logs")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Paso 1: Ejecutar el pipeline y cargar los datos
X_train, X_val, X_test, y_train, y_val, y_test = data_pipeline()

# Paso 2: Ajustar el índice de y_train
if not y_train.index.is_monotonic_increasing:
    print("Ordenando el índice de y_train...")
    y_train = y_train.sort_index()

if y_train.index.freq is None:
    print("Agrupando y asignando frecuencia semanal ('W') al índice...")
    y_train = y_train.resample('W').sum()

# Filtrar columnas numéricas y agrupar X_train, X_val, X_test
X_train_numeric = X_train.select_dtypes(include=["number"])
X_val_numeric = X_val.select_dtypes(include=["number"])
X_test_numeric = X_test.select_dtypes(include=["number"])

X_train = X_train_numeric.resample('W').sum()
X_val = X_val_numeric.resample('W').sum()
X_test = X_test_numeric.resample('W').sum()

# Paso 3: Definir las variables exógenas
exog_columns = [
    "imacec_general", "imacec_comercio", "imacec_no_minero", "ee_comercio",
    "imce_general", "imce_comercio", "icc", "ine_alimentos", "ine_supermercados",
    "tpm", "trimestre", "pib", "es_festivo", "tavg", "tmin", "tmax", "cyber_monday", "black_friday"
]

# Filtrar las columnas que realmente existen en X_train
exog_columns = [col for col in exog_columns if col in X_train.columns]
print("Variables exógenas utilizadas:", exog_columns)

### Entrenamiento y evaluación de modelos ###

# **Modelo sin variables exógenas**
print("\nEntrenando modelo SARIMAX sin variables exógenas...")
sarimax_no_exog = SARIMAXModel(order=(0, 1, 1), seasonal_order=(0, 1, 1, 52))
sarimax_no_exog.fit(y_train)

# Generar predicciones
forecast_no_exog = sarimax_no_exog.predict(steps=len(y_val))
mae_no_exog, rmse_no_exog = sarimax_no_exog.evaluate(y_val, forecast_no_exog)
print(f"\nModelo sin variables exógenas: MAE = {mae_no_exog}, RMSE = {rmse_no_exog}")

# Guardar resultados del modelo sin variables exógenas
sarimax_no_exog.save_model(os.path.join(MODELS_DIR, "sarimax_no_exog.pkl"))
sarimax_no_exog.save_plot(
    y_train=y_train,
    forecast=forecast_no_exog,
    steps=52,  # Limitar a 52 semanas
    file_path=os.path.join(PLOTS_DIR, "sarimax_no_exog_forecast.png")
)
sarimax_no_exog.save_metrics(
    mae=mae_no_exog, rmse=rmse_no_exog,
    save_path=os.path.join(METRICS_DIR, "sarimax_no_exog_metrics.json")
)

# **Modelo con variables exógenas**
print("\nEntrenando modelo SARIMAX con variables exógenas...")
sarimax_with_exog = SARIMAXModel(order=(0, 1, 1), seasonal_order=(0, 1, 1, 52))
sarimax_with_exog.fit(y_train, exog=X_train[exog_columns])

# Generar predicciones
forecast_with_exog = sarimax_with_exog.predict(steps=len(y_val), exog=X_val[exog_columns])
mae_with_exog, rmse_with_exog = sarimax_with_exog.evaluate(y_val, forecast_with_exog)
print(f"\nModelo con variables exógenas: MAE = {mae_with_exog}, RMSE = {rmse_with_exog}")

# Guardar resultados del modelo con variables exógenas
sarimax_with_exog.save_model(os.path.join(MODELS_DIR, "sarimax_with_exog.pkl"))
sarimax_with_exog.save_plot(
    y_train=y_train,
    forecast=forecast_with_exog,
    steps=52,  # Limitar a 52 semanas
    file_path=os.path.join(PLOTS_DIR, "sarimax_with_exog_forecast.png")
)
sarimax_with_exog.save_metrics(
    mae=mae_with_exog, rmse=rmse_with_exog,
    save_path=os.path.join(METRICS_DIR, "sarimax_with_exog_metrics.json")
)
