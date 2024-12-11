import pandas as pd
from src.data.utils.config import get_config
import matplotlib.pyplot as plt
import dask.dataframe as dd

#from utils.config import get_config

class DataIntegrator:
    def __init__(self, config_mode):
        self.config_mode = config_mode
        self.config_dict = get_config(config_mode)
        self.aggregation_methods = self.config_dict['data']['aggregation_methods']

        self.join_keys = self.config_dict['join_keys']

    def integrate(self, df_sales, df_stock, df_exog):
        """
        Merge sales data, stock data and exogenous data
        """
        # Merge sales and exogenous data
        df_sales_exog = df_sales.merge(df_exog, on='fecha', how='left')
        df_sales_agg = df_sales_exog.groupby(['codigo_producto2', 'fecha']).agg(self.aggregation_methods).reset_index()

        # Merge stock data
        df_stock_agg = df_stock.groupby(['codigo_producto2', 'fecha']).agg({
            'stock_disponible_total': 'sum',
        }).reset_index()

        # date range 
        start_end = {
            'start': df_stock_agg['fecha'].min(),
            'end': df_stock_agg['fecha'].max()
        }
        date_range = pd.date_range(start=start_end['start'], end=start_end['end'], freq='D')
        base_stock = pd.MultiIndex.from_product([df_stock_agg['codigo_producto2'].unique(), date_range], names=['codigo_producto2', 'fecha']).to_frame(index=False)
        df_consolidated = base_stock.merge(df_stock_agg, on=['codigo_producto2', 'fecha'], how='left')
        df_consolidated = df_consolidated.rename(columns={'codigo_producto2': 'codigo_producto_stock'})
        print('ya renombre')
        df_consolidated.fillna({'stock_disponible_total': 0}, inplace=True)
        print('ya llene')

        print('voy a mergear')
        df = self._merge_by_chunks(df_sales_agg, df_consolidated)
        print('hola')

        df['fecha'] = pd.to_datetime(df['fecha'])
        df['semana'] = df['fecha'].dt.to_period('W').astype(str)

        stock_mean_weekly = df.groupby(['codigo_producto2', 'semana'])['stock_disponible_total'].mean().reset_index()
        stock_mean_weekly = stock_mean_weekly.rename(columns={'stock_disponible_total': 'stock_media_semanal'})

        df = df.merge(stock_mean_weekly, on=['codigo_producto2', 'semana'], how='left')
        df['stock_disponible_total'] = df['stock_disponible_total'].fillna(df['stock_media_semanal'])
        df = df.drop(columns=['stock_media_semanal'])
        df = df[df['fecha'].dt.year != 2020]
        print('no me cai')

        return df
    
    def _merge_by_chunks(self, df1, df2):
        """
        Merge dataframes by chunks
        """
        df_sales_maestro_dd = dd.from_pandas(df1, npartitions=10)
        df_consolidated_dd = dd.from_pandas(df2, npartitions=10)

        # Realizar el merge con Dask
        df_merged = dd.merge(df_sales_maestro_dd, df_consolidated_dd, on='fecha', how='left')

        # Convertir de nuevo a pandas si es necesario
        df_merged = df_merged.compute()
        return df_merged

