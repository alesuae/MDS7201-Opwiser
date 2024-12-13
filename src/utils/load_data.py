import pandas as pd
from typing import Tuple

def load_data(data_path: str) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, 
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
]:
    """
    Carga los datos procesados y particionados para entrenar el modelo.

    Args:
        data_path (str): Ruta al dataset procesado.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, 
              pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
              Conjuntos de entrenamiento y prueba.
    """
    # Cargar los datos normales como DataFrame
    X_train = pd.read_csv(f'{data_path}/X_train.csv')
    X_test = pd.read_csv(f'{data_path}/X_test.csv')
    y_train = pd.read_csv(f'{data_path}/y_train.csv')
    y_test = pd.read_csv(f'{data_path}/y_test.csv')

    # Cargar los datos temporales
    X_train_temp = pd.read_csv(f'{data_path}/X_train_temp.csv', index_col='fecha', parse_dates=['fecha'])
    X_test_temp = pd.read_csv(f'{data_path}/X_test_temp.csv', index_col='fecha', parse_dates=['fecha'])
    y_train_temp = pd.read_csv(f'{data_path}/y_train_temp.csv', index_col='fecha', parse_dates=['fecha']).squeeze("columns")
    y_test_temp = pd.read_csv(f'{data_path}/y_test_temp.csv', index_col='fecha', parse_dates=['fecha']).squeeze("columns")

    return X_train, X_test, y_train, y_test, X_train_temp, X_test_temp, y_train_temp, y_test_temp
