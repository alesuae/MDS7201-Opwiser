import pandas as pd
from utils.data_downloader import download_data_from_api
from base.progress_bar import ProgressBar

class ExogenousDataExtractor:
    def __init__(self, config_mode):
        self.config_filepath = config_mode
        self.progress_bar = ProgressBar()

    def fetch_data(self):
        """
        Uses the download_data_from_api function from utils to fetch data.
        """
        self.progress_bar.update_total_steps(1)
        self.progress_bar.log("Requesting data from API...")

        df_monthly, df_quarterly = download_data_from_api(self.config_mode)
        
        self.progress_bar.check()
        self.progress_bar.close()
        
        return df_monthly, df_quarterly

    def preprocess_data(self, df, date_column):
        """
        Performs initial data preprocessing.
        """
        self.progress_bar.update_total_steps(1)
        self.progress_bar.log(f"Converting {date_column} to date format")
        
        df[date_column] = pd.to_datetime(df[date_column])
        self.progress_bar.check()
        self.progress_bar.close()
        
        return df
