from base.base_dataset_maker import BaseDatasetMaker
import pandas as pd

class SalesDatasetMaker(BaseDatasetMaker):
    def load_data(self):
        """Load data from a CSV file or any other source."""
        self.data = pd.read_csv(self.source_path)
        print("Sales data loaded.")

    def clean_data(self):
        """Clean the data by removing null values and duplicate rows."""
        self.data.dropna(inplace=True)
        self.data.drop_duplicates(inplace=True)
        print("Sales data cleaned.")

    def transform_data(self):
        """Transform columns, perform aggregations, or other modifications."""
        # Example transformation: convert dates to a single format
        self.data['fecha'] = pd.to_datetime(self.data['fecha'])
        print("Sales data transformed.")

    def _load_csv(self, path):
        """Private method to load a CSV file."""
        return pd.read_csv(path)

