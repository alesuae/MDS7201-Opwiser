from src.data.data_preprocessing.prepare_data import DataPreparer
from src.data.data_preprocessing.data_splitter import DataSplitter
from src.models.lightgbm_model import LightGBMModel
from src.models.arima_model import ARIMAModel
from utils.yaml_reader import load_yaml

# Cargar configuraciones
model_config = load_yaml("model.config.yaml")

# Seleccionar modelo
model_name = "LightGBM"  # Cambia esto dinámicamente según el flujo
model_settings = model_config['models'][model_name]

# Preparar datos
preparer = DataPreparer(config=model_settings['preprocesamiento'])
df_prepared = preparer.prepare(df_combined)

# Dividir datos
splitter = DataSplitter(config=model_settings)
splits = splitter.split(df_prepared, target='venta_total_neto')

# Entrenar modelo
if model_name == "LightGBM":
    model = LightGBMModel(params=model_settings['hyperparameters'])
elif model_name == "ARIMA":
    model = ARIMAModel(params=model_settings['hyperparameters'])

# Usar los splits para entrenar/validar
model.train(*splits[:2])  # Entrenamiento con X_train, y_train
