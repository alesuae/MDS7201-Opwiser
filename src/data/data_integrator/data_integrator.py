import pandas as pd

# TODO: implement query to merge data

class DataIntegrator:
    def __init__(self, datasets):
        """
        datasets: dictionary containing the datasets to integrate.
        Example: {'sales': sales_df, 'stock': stock_df}
        """
        self.datasets = datasets
        self.integrated_data = None

    def merge_datasets(self, key_column='product_code', how='inner'):
        """
        Merge the datasets in the `datasets` dictionary using `key_column`.
        """
        data_frames = list(self.datasets.values())
        integrated_df = data_frames[0]
        
        for df in data_frames[1:]:
            integrated_df = integrated_df.merge(df, on=key_column, how=how)
        
        self.integrated_data = integrated_df

    def add_exogenous_variables(self, exogenous_data, on_column='date'):
        """
        Add exogenous variables to `integrated_data`.
        """
        if self.integrated_data is None:
            raise ValueError("Datasets have not been merged. Execute `merge_datasets` first.")
        
        self.integrated_data = self.integrated_data.merge(exogenous_data, on=on_column, how='left')

    def get_integrated_data(self):
        """Return the integrated dataframe with exogenous variables."""
        return self.integrated_data
