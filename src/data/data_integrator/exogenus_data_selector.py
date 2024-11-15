import pandas as pd

# TODO: load config dict

class ExogenousDataSelector:
    def __init__(self, exogenous_data: pd.DataFrame) -> pd.DataFrame:
        self.exogenous_data = exogenous_data

    def select_variables(self, types: list):
        """
        Filters the exogenous data by the specified types (e.g., 'weather', 'macroeconomic', 'holidays').

        :param types: List of categories to filter by, e.g., ["weather", "macroeconomic", "holidays"].
        :return: Filtered DataFrame.
        """
        # Assuming the exogenous_data has a 'type' column to identify the variable category
        if 'type' not in self.exogenous_data.columns:
            raise ValueError("The exogenous data must contain a 'type' column to filter by category.")

        filtered_data = self.exogenous_data[self.exogenous_data['type'].isin(types)]
        return filtered_data
