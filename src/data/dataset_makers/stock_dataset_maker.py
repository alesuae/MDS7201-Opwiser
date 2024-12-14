from src.data.base.base_dataset_maker import BaseDatasetMaker
from src.data.base.progress_bar import ProgressBar
from src.data.utils.config import get_config

import pandas as pd
import polars as pl

class StockDatasetMaker(BaseDatasetMaker):
    def __init__(self, config_mode: str):
        self.config_mode = config_mode
        self.progress_bar = ProgressBar()

        self.config_dict = get_config(self.config_mode)
        self.source_path = self.config_dict['stock']["path"]

        self.numeric_variables = self.config_dict["stock"]["numeric_vars"].values()
        self.sku = self.config_dict["sku"]

    def load_data(self):
        self.progress_bar.update_total_steps(1)
        self.progress_bar.log("Loading stock data...")
        self.data = pl.read_csv(source=self.source_path, separator=';')
        self.data = self.data.to_pandas()
        self.progress_bar.check()
        self.progress_bar.close()

    def clean_data(self):
        variables = list(self.numeric_variables)

        self.progress_bar.log("Transforming stock data to numeric")
        for idx in range(len(variables)):
            self.data[variables[idx]] = pd.to_numeric(self.data[variables[idx]].str.replace(',', ''), errors='raise')
        self.progress_bar.check()
        self.progress_bar.close()
