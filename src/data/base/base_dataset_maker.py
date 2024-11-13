from abc import ABC, abstractmethod
import pandas as pd
from base.progress_bar import ProgressBar

class BaseDatasetMaker(ABC):
    def __init__(self, source_path: str):
        self.source_path = source_path
        self.data = None
        self.progress_bar = ProgressBar()

    @abstractmethod
    def load_data(self):
        """
        Method to load data from the specified source

        Args:
            None.
        Returns:
            None.
        """
        pass

    def basic_cleaning(self):
        """
        Basic cleaning for all datasets coming from R

        Args:
            None.
        Returns:
            None.
        """
        self.progress_bar.update_total_steps(3)
        self.progress_bar.log("Starting basic cleaning...")

        self.progress_bar.log("Removing spaces in column names")
        self.data.columns = self.data.columns.str.strip()  # Remove spaces in column names
        self.progress_bar.check()

        self.progress_bar.log("Removing completely empty rows")
        self.data.dropna(how='all', inplace=True)  # Remove completely empty rows
        self.progress_bar.check()

        self.progress_bar.log("Removing spaces in string values")
        self.data = self.data.applymap(lambda x: x.strip() if isinstance(x, str) else x)  # Remove spaces in strings
        self.progress_bar.check()

        self.progress_bar.close()

    def convert_dates(self, date_column:str):
        """
        Converts date columns from string to datetime format

        Args:
            None.
        Returns:
            None.
        """
        self.progress_bar.update_total_steps(1)
        self.progress_bar.log(f"Converting {date_column} to date format")
        self.data[date_column] = pd.to_datetime(self.data[date_column], errors='coerce')
        self.progress_bar.check()
        self.progress_bar.close()

    def get_data(self):
        """"
        Returns dataset
        """
        return self.data
