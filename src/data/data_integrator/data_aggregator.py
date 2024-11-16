import pandas as pd
from utils.config import get_config


# TODO: remove hardcoded data!!

class DataAggregator:
    def __init__(self, config_mode):
        self.config_mode = config_mode
        self.config_dict = get_config(self.config_mode)

        self.aggregation_level = self.config_dict['aggregation_level']
        self.aggregation_methods = self.config_dict['aggregation_methods']

    # TODO: add fill missing stock values with 0 !!!!
    def aggregate(self, df):
        """
        Agrega el dataset según el nivel y métodos configurados.
        """
        if self.aggregation_level == 'monthly':
            df['mes'] = pd.to_datetime(df['fecha']).dt.to_period('M')
            groupby_cols = ['mes', 'codigo_producto2']
        elif self.aggregation_level == 'weekly':
            df['semana'] = pd.to_datetime(df['fecha']).dt.to_period('W')
            groupby_cols = ['semana', 'codigo_producto2']
        else:
            raise ValueError(f"Aggregation level '{self.aggregation_level}' not found.")

        aggregated_df = df.groupby(groupby_cols).agg(self.aggregation_methods).reset_index()

        return aggregated_df
