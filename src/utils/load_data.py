import pandas as pd
from typing import Tuple

def load_data(data_path:str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga los datos procesados y particionados para entrenar el modelo.

    Args:
        data_path (str): Ruta al dataset procesado.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: Conjuntos de entrenamiento y prueba.
    """
    X_train = pd.read_csv(f'{data_path}/X_train.csv')
    X_test = pd.read_csv(f'{data_path}/X_test.csv')
    y_train = pd.read_csv(f'{data_path}/y_train.csv')
    y_test = pd.read_csv(f'{data_path}/y_test.csv')
    return X_train, X_test, y_train, y_test

def load_semana(data_path:str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X_train_semana = pd.read_csv(f'{data_path}/X_train_semana.csv')
    X_test_semana = pd.read_csv(f'{data_path}/X_test_semana.csv')
    return X_train_semana, X_test_semana
