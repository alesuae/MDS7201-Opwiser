from sklearn.model_selection import train_test_split, TimeSeriesSplit
from utils.config import get_config

class DataSplitter:
    def __init__(self, config_mode):
        """
        Configuration for data splitting.
        """
        self.config_dict = get_config(config_mode)

        self.division = self.config_dict['division_type'] 
        self.cv_splits = self.config_dict['cv_splits'] 
        self.test_size = self.config_dict['test_size'] 
        self.validation_size = self.config_dict['val_size'] 

    def split(self, df, target):
        """
        Split the dataset based on the configuration.
        """
        if self.division == 'train_val_test':
            return self.train_val_test_split(df, target)
        elif self.division == 'cross_validation':
            return self.cross_validate(df, target)
        else:
            raise ValueError(f"Unsupported division type '{self.division}'.")

    def train_val_test_split(self, df, target):
        """
        Split into train, validation, and test sets.
        """
        X = df.drop(columns=[target])
        y = df[target]

        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=self.test_size, shuffle=False)
        val_size = self.validation_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, shuffle=False)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def cross_validate(self, df, target):
        """
        Split for time series cross-validation.
        """
        X = df.drop(columns=[target])
        y = df[target]

        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        return list(tscv.split(X, y))
