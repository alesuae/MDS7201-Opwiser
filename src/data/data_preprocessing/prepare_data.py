import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.pipeline import Pipeline

class DataPreparer:
    def __init__(self, config):
        """
        Configuration based on the model from YAML.
        """
        self.impute_method = config.get('impute_method', 'mean')
        self.scaler = config.get('scaler', None)
        self.transformer = config.get('power_transformer', None)

    def prepare(self, df):
        """
        Prepare the dataset according to the current configuration.
        """
        df = self.impute_missing_values(df)

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
            return df.fillna(df.mean())
        elif self.impute_method == 'median':
            return df.fillna(df.median())
        elif self.impute_method == 'drop':
            return df.dropna()
        else:
            raise ValueError(f"Impute method '{self.impute_method}' not supported.")

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
        return pd.DataFrame(pipeline.fit_transform(df), columns=df.columns)
