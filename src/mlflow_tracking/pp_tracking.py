from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow

from src.data.data_preprocessing.prepare_data import DataPreparer


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
        X = data_df.drop(columns=['venta_total_neto']) 
        y = data_df['venta_total_neto']  
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        X_train.to_csv('data/splits/X_train.csv', index=False)
        y_train.to_csv('data/splits/y_train.csv', index=False)
        X_test.to_csv('data/splits/X_test.csv', index=False)
        y_test.to_csv('data/splits/y_test.csv', index=False)

        # Registrar métricas y artefactos
        mlflow.log_param("rows_train", X_train.shape[0])
        mlflow.log_param("columns_train", X_train.shape[1])
        mlflow.log_param("rows_test", X_test.shape[0])
        mlflow.log_param("columns_test", X_test.shape[1])
        mlflow.log_artifact(output_path, artifact_path="data/splits")
        print("Partición rastreada en MLFlow.")

def log_temporal_splitter(data_df: pd.DataFrame, target: str, output_path: str) -> pd.DataFrame:
    """
    Rastrea el paso de partición de datos de modelos temporales (ARIMA, Prophet) en MLFlow.

    Args:
        data_df (pd.DataFrame): DataFrame con los datos a particionar.
        target (str): Nombre de la columna objetivo.
        output_path (str): Ruta al dataset particionado.
    """
    with mlflow.start_run(run_name="Splitting"):
        # Validar y convertir 'fecha' a datetime
        if 'fecha' not in data_df.columns:
            raise ValueError("El conjunto de datos debe tener una columna llamada 'fecha'.")
        
        print("Validando y configurando 'fecha'...")
        data_df['fecha'] = pd.to_datetime(data_df['fecha'], errors='coerce')
        if data_df['fecha'].isnull().any():
            raise ValueError("Se encontraron fechas no válidas en la columna 'fecha' después de la conversión.")
        
        # División en X (predictoras) e y (objetivo)
        X = data_df.drop(columns=[target]).fillna(0)  # Rellenar valores nulos en X
        y = data_df[target].fillna(0)  # Rellenar valores nulos en y

        # Dividir en conjuntos de entrenamiento y prueba "normales"
        print("Dividiendo conjuntos de datos 'normales'...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

        # Agrupar por día para ARIMA
        print("Agrupando conjuntos temporales por día...")
        X_train_temp = X_train.groupby('fecha').sum()  # Agrupar por día
        X_test_temp = X_test.groupby('fecha').sum()
        y_train_temp = y_train.groupby(data_df.loc[y_train.index, 'fecha']).sum()
        y_test_temp = y_test.groupby(data_df.loc[y_test.index, 'fecha']).sum()

        # Validar índices y rellenar valores nulos
        for temp_data, name in zip(
            [X_train_temp, X_test_temp, y_train_temp, y_test_temp],
            ["X_train_temp", "X_test_temp", "y_train_temp", "y_test_temp"]
        ):
            print(f"Validando y corrigiendo {name}...")
            temp_data.index = pd.to_datetime(temp_data.index, errors='coerce')
            temp_data = temp_data.sort_index().fillna(0)

        # Guardar particiones en archivos CSV
        print("Guardando datos particionados...")
        X_train.to_csv(f"{output_path}/X_train.csv", index=False)
        y_train.to_csv(f"{output_path}/y_train.csv", index=False)
        X_test.to_csv(f"{output_path}/X_test.csv", index=False)
        y_test.to_csv(f"{output_path}/y_test.csv", index=False)

        # Guardar también las versiones temporales
        X_train_temp.to_csv(f"{output_path}/X_train_temp.csv", index=True)
        X_test_temp.to_csv(f"{output_path}/X_test_temp.csv", index=True)
        y_train_temp.to_csv(f"{output_path}/y_train_temp.csv", index=True)
        y_test_temp.to_csv(f"{output_path}/y_test_temp.csv", index=True)

        # Registrar métricas y artefactos en MLFlow
        mlflow.log_param("rows_train", X_train.shape[0])
        mlflow.log_param("columns_train", X_train.shape[1])
        mlflow.log_param("rows_test", X_test.shape[0])
        mlflow.log_param("columns_test", X_test.shape[1])
        mlflow.log_param("rows_train_temp", X_train_temp.shape[0])
        mlflow.log_param("columns_train_temp", X_train_temp.shape[1])
        mlflow.log_param("rows_test_temp", X_test_temp.shape[0])
        mlflow.log_param("columns_test_temp", X_test_temp.shape[1])
        mlflow.log_artifact(output_path, artifact_path="data/splits")

        print("Partición rastreada en MLFlow.")
