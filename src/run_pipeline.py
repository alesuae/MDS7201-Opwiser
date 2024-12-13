from src.data.data_preprocessing.prepare_data import DataPreparer
from src.data.data_preprocessing.data_splitter import DataSplitter
# TODO: HACERLOOOOOOOAAAAAA
from src.models.lightgbm_model import LightGBMModel
from src.models.arima_model import ARIMAModel
from src.models.utils.model_config import get_config

# Cargar configuraciones
model_config = get_config('model')

# Seleccionar modelo
model_name = "LightGBM"  # Cambia esto dinámicamente según el flujo
model_settings = model_config['models'][model_name]

from src.data.main import data_pipeline
X_train, X_val, X_test, y_train, y_val, y_test = data_pipeline()

# Entrenar modelo
if model_name == "LightGBM":
    model = LightGBMModel(params=model_settings['hyperparameters'])
elif model_name == "ARIMA":
    model = ARIMAModel(params=model_settings['hyperparameters'])

# Usar los splits para entrenar/validar
model.train(X_train, y_train)  # Entrenamiento con X_train, y_train
