from base.base_dataset_maker import BaseDatasetMaker
import pandas as pd


class StockDatasetMaker(BaseDatasetMaker):
    def load_data(self):
        self.data = pd.read_csv(self.source_path)
        print("Stock data loaded.")

    def clean_data(self):
        # TODO: implement data imputation of missing values
        self.data.fillna(0, inplace=True)
        print("Stock data cleaned.")

    def transform_data(self):
        # TODO: check units of measurement
        # Example transformation to unify units of measurement, if necessary
        print("Stock data transformed.")
