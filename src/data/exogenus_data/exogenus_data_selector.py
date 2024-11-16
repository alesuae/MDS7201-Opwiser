import pandas as pd

class ExogenousDataSelector:
    def __init__(self, config):
        """
        Inicializa el selector con configuraciones desde el YAML.
        """
        self.variables = config['variables']  # Lista de variables exógenas relevantes

    def select(self, df_exogenous: pd.DataFrame):
        """
        Filtra el dataset para incluir solo las variables seleccionadas.
        """
        # Siempre incluir la columna de fecha para integración posterior
        selected_columns = ['fecha'] + self.variables
        return df_exogenous[selected_columns]
