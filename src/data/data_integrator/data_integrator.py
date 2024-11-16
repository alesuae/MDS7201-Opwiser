import pandas as pd
from utils.config import get_config

class DataIntegrator:
    def __init__(self, config_mode):
        self.config_mode = config_mode
        self.config_dict = get_config(config_mode)

        self.join_keys = self.config_dict['join_keys']

    def integrate(self, df_sales, df_stock, df_exog):
        """
        Merge sales data, stock data and exogenous data
        """
        # Merge sales and exogenous data
        df_combined = df_sales.merge(df_exog, on='fecha', how='left')

        # Merge stock data
        # TODO: Check sql query for stock imputation
        df_combined = df_combined.merge(df_stock, on=self.join_keys, how='left')
        return df_combined

