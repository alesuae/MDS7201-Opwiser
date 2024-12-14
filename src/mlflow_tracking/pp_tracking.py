from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow

from src.data.data_preprocessing.prepare_data import DataPreparer
from src.utils.config import get_config

config_dict = get_config('data')
avg_date = config_dict['avg_date']

def log_preprocessing(data:pd.DataFrame, output_path:str) -> pd.DataFrame:
    """
    Rastrea el paso de preprocesamiento en MLFlow.

    Args:
        data_path (str): Ruta al dataset original.
        output_path (str): Ruta al dataset procesado.
    """
    with mlflow.start_run(run_name="Preprocessing"):
        # Cargar y procesar datos
        preparer = DataPreparer(config_mode='data')
        processed_data = preparer.prepare(data)

        processed_data.to_csv(output_path, index=False)

        # Registrar métricas y artefactos
        mlflow.log_param("rows_before", data.shape[0])
        mlflow.log_param("columns_before", data.shape[1])
        mlflow.log_param("rows_after", processed_data.shape[0])
        mlflow.log_param("columns_after", processed_data.shape[1])
        mlflow.log_artifact(output_path, artifact_path="data/processed")
        print("Preprocesamiento rastreado en MLFlow.")

def log_splitter(data_df:pd.DataFrame, output_path:str) -> pd.DataFrame:
    """
    Rastrea el paso de partición de datos de modelos ML en MLFlow.

    Args:
        data_df (pd.DataFrame): DataFrame con los datos a particionar.
        output_path (str): Ruta al dataset particionado.
    """
    with mlflow.start_run(run_name="Splitting"):
        # Dividir datos
        #y = data_df['venta_total_neto']
        #X = data_df.drop(columns=['venta_total_neto']) 
          
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Dividir respetando la secuencia temporal
        train_data = data_df[data_df['fecha'] < avg_date]
        test_data = data_df[data_df['fecha'] >= avg_date]

        X_train = train_data.drop(columns=['venta_total_neto'])
        y_train = train_data['venta_total_neto']
        X_test = test_data.drop(columns=['venta_total_neto'])
        y_test = test_data['venta_total_neto']

        X_train_semana = X_train["semana"].values
        X_train_categoria = X_train["categoria_2"].values
        X_train_fecha = X_train["fecha"].values
        X_train = X_train.drop(columns=["semana", "categoria_2", "fecha"])
        
        X_test_semana = X_test["semana"].values
        X_test_categoria = X_test["categoria_2"].values
        X_test_fecha = X_test["fecha"].values
        X_test = X_test.drop(columns=["semana", "categoria_2", "fecha"])    
        
        original_agg_train = pd.DataFrame({
            "X_train_semana": X_train_semana,
            "X_train_categoria": X_train_categoria,  
            "X_train_fecha": X_train_fecha
        })
        original_agg_test = pd.DataFrame({
            "X_test_semana": X_test_semana,
            "X_test_categoria": X_test_categoria,
            "X_test_fecha": X_test_fecha   
        })

        X_train.to_csv('data/splits/X_train.csv', index=False)
        y_train.to_csv('data/splits/y_train.csv', index=False)
        X_test.to_csv('data/splits/X_test.csv', index=False)
        y_test.to_csv('data/splits/y_test.csv', index=False)
        original_agg_train.to_csv('data/splits/original_agg_train.csv', index=False)
        original_agg_test.to_csv('data/splits/original_agg_test.csv', index=False)

        # Registrar métricas y artefactos
        mlflow.log_param("rows_train", X_train.shape[0])
        mlflow.log_param("columns_train", X_train.shape[1])
        mlflow.log_param("rows_test", X_test.shape[0])
        mlflow.log_param("columns_test", X_test.shape[1])
        mlflow.log_artifact(output_path, artifact_path="data/splits")
        print("Partición rastreada en MLFlow.")

def log_temporal_splitter(data_df:pd.DataFrame, target:str, output_path:str) -> pd.DataFrame:
    """
    Rastrea el paso de partición de datos de modelos temporales (arima, prophet) en MLFlow.

    Args:
        data_df (pd.DataFrame): DataFrame con los datos a particionar.
        output_path (str): Ruta al dataset particionado.
    """
    with mlflow.start_run(run_name="Splitting"):
        #  Ensure the 'fecha' column is properly set as the DatetimeIndex
        if 'fecha' not in data_df.columns:
            raise ValueError("El conjunto de datos agregado debe tener una columna llamada 'fecha'.")
        
        data_df['fecha'] = pd.to_datetime(data_df['fecha'], errors='coerce')

        if data_df['fecha'].isnull().any():
            raise ValueError("Se encontraron fechas no válidas en la columna 'fecha' después de la conversión.")
        
        # Aseguramos que 'fecha' sea el índice temporal
        data_df.set_index('fecha', inplace=True, drop=False)
        # Verificamos que el índice esté ordenado y con frecuencia asignada
        if not data_df.index.is_monotonic_increasing:
            data_df = data_df.sort_index()

        X = data_df.drop(columns=[target])
        y = data_df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
        X_train_semana = X_train["semana"].values
        X_train_categoria = X_train["categoria_2"].values
        X_train = X_train.drop(columns=["semana", "categoria_2"])
        
        X_test_semana = X_test["semana"].values
        X_test_categoria = X_test["categoria_2"].values
        X_test = X_test.drop(columns=["semana", "categoria_2"])  
        
        original_agg_train = pd.DataFrame({
            "X_train_semana": X_train_semana,
            "X_train_categoria": X_train_categoria,  
        })
        original_agg_test = pd.DataFrame({
            "X_test_semana": X_test_semana,
            "X_test_categoria": X_test_categoria   
        })

        X_train.to_csv('data/splits/X_train.csv', index=False)
        y_train.to_csv('data/splits/y_train.csv', index=False)
        X_test.to_csv('data/splits/X_test.csv', index=False)
        y_test.to_csv('data/splits/y_test.csv', index=False)
        original_agg_train.to_csv('data/splits/original_agg_train.csv', index=False)
        original_agg_test.to_csv('data/splits/original_agg_test.csv', index=False)

        # Registrar métricas y artefactos
        mlflow.log_param("rows_train", X_train.shape[0])
        mlflow.log_param("columns_train", X_train.shape[1])
        mlflow.log_param("rows_test", X_test.shape[0])
        mlflow.log_param("columns_test", X_test.shape[1])
        mlflow.log_artifact(output_path, artifact_path="data/splits")
        print("Partición rastreada en MLFlow.")





