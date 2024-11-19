import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
#from utils.config import get_config

from src.data.utils.config import get_config
import numpy as np

class DataPreparer:
    def __init__(self, config_mode):
        """
        Configuration based on the model from YAML.
        """
        self.config_dict = get_config(config_mode)
        self.impute_method = self.config_dict['impute_method'] 
        self.scaler = self.config_dict['scaler']
        self.transformer = self.config_dict['power_transformer'] 
        self.apply_log_transform = self.config_dict['log_transform'] 
        self.categorical_encoding = self.config_dict['categorical_encoding'] 

    def prepare(self, df):
        """
        Prepare the dataset according to the current configuration.
        """
        df = self.impute_missing_values(df)

        # Handle categorical variables
        df = self.handle_categorical_variables(df)

        # Apply log transform if enabled
        df = self.log_transform(df)

        # Scale and transform only if configured
        if self.scaler or self.transformer:
            df = self.scale_and_transform(df)

        return df

    def impute_missing_values(self, df):
        """
        Handle missing values based on configuration.
        """
        if self.impute_method == 'zero':
            return df.fillna(0)
        elif self.impute_method == 'mean':
            return df.fillna(df.mean(numeric_only=True))
        elif self.impute_method == 'median':
            return df.fillna(df.median(numeric_only=True))
        elif self.impute_method == 'drop':
            return df.dropna()
        else:
            raise ValueError(f"Impute method '{self.impute_method}' not supported.")

    def handle_categorical_variables(self, df):
        """
        Identify and encode categorical variables.
        """
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not categorical_cols:
            return df

        print("Categorical variables detected:", categorical_cols)

        if self.categorical_encoding == 'onehot':
            print("Applying One-Hot Encoding...")
            df = pd.get_dummies(df, columns=categorical_cols)
        elif self.categorical_encoding == 'label':
            print("Applying Label Encoding...")
            for col in categorical_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        else:
            raise ValueError(f"Categorical encoding method '{self.categorical_encoding}' not supported.")

        return df

    def log_transform(self, df):
        """
        Apply log transformation to all positive columns.
        """
        print("Applying log transformation...")
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64'] and (df[col] > 0).all():
                df[col] = np.log1p(df[col])  # log1p handles log(1 + x)
            else:
                print(f"Skipping log transform for column '{col}' (non-numeric or non-positive values).")
        return df

    def scale_and_transform(self, df):
        """
        Apply scaling and transformation.
        """
        steps = []

        if self.scaler == 'standard':
            steps.append(('scaler', StandardScaler()))
        elif self.scaler == 'minmax':
            steps.append(('scaler', MinMaxScaler()))

        if self.transformer:
            steps.append(('transformer', PowerTransformer(method=self.transformer)))

        pipeline = Pipeline(steps)
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = pipeline.fit_transform(df[numeric_cols])

        return df
