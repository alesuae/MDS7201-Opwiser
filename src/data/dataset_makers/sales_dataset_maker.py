from src.data.base.base_dataset_maker import BaseDatasetMaker
from src.data.base.progress_bar import ProgressBar
from src.data.utils.config import get_config
from src.data.utils.sku_merger import merge_by_sku



#from base.base_dataset_maker import BaseDatasetMaker
#from base.progress_bar import ProgressBar
#from utils.config import get_config
#from utils.sku_merger import merge_by_sku
import pandas as pd
import polars as pl

class SalesDatasetMaker(BaseDatasetMaker):
    def __init__(self, config_mode: str):
        self.config_mode = config_mode
        self.progress_bar = ProgressBar()

        self.config_dict = get_config(self.config_mode)
        self.source_path = self.config_dict['sales']["path"]
        self.maestro_path = self.config_dict['sales']["maestro_path"]

        self.numeric_variables = self.config_dict["sales"]["numeric_vars"].values()
        self.sku = self.config_dict["sku"]

    def load_data(self): 
        """Load data from a CSV file or any other source."""
        self.progress_bar.update_total_steps(1)
        self.progress_bar.log("Loading sales data and maestro data...")
        self.data = pl.read_csv(source=self.source_path, separator=';')
        self.maestro_data = pl.read_csv(source=self.maestro_path, separator=';')
        self.data = self.data.to_pandas()
        self.maestro_data = self.maestro_data.to_pandas()
        self.progress_bar.check()
        self.progress_bar.close()


    def clean_data(self):
        variables = list(self.numeric_variables)

        # Delete missing data (SKU)
        self.progress_bar.update_total_steps(5)
        self.progress_bar.log("Starting data transformation...")
        self.progress_bar.log("Removing missing data by sku")
        self.data = self.data.dropna(subset=[self.sku])
        self.progress_bar.check()
        
        # Transform numeric data to float
        self.progress_bar.log("Transforming sales data to numeric")
        for idx in range(len(variables)):
            self.data[variables[idx]] = pd.to_numeric(self.data[variables[idx]].str.replace(',', ''), errors='raise')
        self.progress_bar.check()
        
        # Merge with maestro data
        self.progress_bar.log("Merging sales data with maestro data")
        self.data = merge_by_sku(self.sku, self.data, self.maestro_data)
        self.progress_bar.check()

        # Add date features
        self.progress_bar.update_total_steps(1)
        self.progress_bar.log("Adding date features...")
        self.data['year_month'] = self.data['fecha'].dt.to_period('M')
        self.data['quarter'] = self.data['fecha'].dt.to_period('Q')
        self.progress_bar.check()
        self.progress_bar.close()
