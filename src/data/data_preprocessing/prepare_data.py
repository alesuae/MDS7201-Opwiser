import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    PowerTransformer,
    OneHotEncoder,
    LabelEncoder,
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.data.utils.config import get_config
from src.data.utils.transformers_d import DateTransformer, SemanaTransformer
import numpy as np


class DataPreparer:
    def __init__(self, config_mode):
        """
        Configuration based on the model from YAML.
        """
        self.config_dict = get_config(config_mode)
        self.impute_method = self.config_dict["impute_method"]
        self.scaler = self.config_dict["scaler"]
        self.transformer = self.config_dict["power_transformer"]
        self.apply_log_transform = self.config_dict["log_transform"]
        self.categorical_encoding = self.config_dict["categorical_encoding"]

        self.temporal_data = self.config_dict["temporal_data"]

    def prepare(self, df):
        """
        Prepare the dataset according to the current configuration.
        """
        original_columns = df.columns
        df = self.impute_missing_values(df)

        # Apply log transform if enabled
        df = self.log_transform(df)

        if self.temporal_data:
            timestamp_columns = df.select_dtypes(include=["datetime64"]).columns
            date_column_original = df[timestamp_columns]
            df = df.drop(columns=timestamp_columns)
            timestamp_columns = df.select_dtypes(include=["datetime64"]).columns
        else:
            timestamp_columns = df.select_dtypes(include=["datetime64"]).columns

        numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns
        df_ns = df.copy()
        df_ns = df_ns.drop(columns=['semana'])
        categorical_columns = df_ns.select_dtypes(include=["object", "category", "string"]).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ("date", DateTransformer(), timestamp_columns),
                ("week", SemanaTransformer(), ['semana']),
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("scaler", StandardScaler()),
                            ("transformer", PowerTransformer(method=self.transformer)),
                        ]
                    ),
                    numerical_columns,
                ),
                (
                    "cat",
                    OneHotEncoder(),
                    categorical_columns,
                ),
            ],
            remainder="passthrough",  # Dejar columnas no especificadas
        ).fit(df)

        date_column_names = preprocessor.named_transformers_["date"].get_feature_names_out()
        week_column_names = preprocessor.named_transformers_["week"].get_feature_names_out()
        num_column_names = preprocessor.named_transformers_["num"].steps[0][1].get_feature_names_out()
        cat_column_names = preprocessor.named_transformers_["cat"].get_feature_names_out()

        transformed_column_names = np.append(
            date_column_names, week_column_names
        )
        transformed_column_names = np.append(
            transformed_column_names, num_column_names
        )
        transformed_column_names = np.append(
            transformed_column_names, cat_column_names
        )
        preprocessor_pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
        processed_data = preprocessor_pipeline.fit_transform(df)
        print(processed_data)

        processed_data_df = pd.DataFrame(
            processed_data, columns=transformed_column_names
        )

        if self.temporal_data:
            processed_data_df['fecha'] = date_column_original

        return processed_data_df

    def impute_missing_values(self, df):
        """
        Handle missing values based on configuration.
        """
        if self.impute_method == "zero":
            return df.fillna(0)
        elif self.impute_method == "mean":
            return df.fillna(df.mean(numeric_only=True))
        elif self.impute_method == "median":
            return df.fillna(df.median(numeric_only=True))
        elif self.impute_method == "drop":
            return df.dropna()
        else:
            raise ValueError(f"Impute method '{self.impute_method}' not supported.")

    def log_transform(self, df):
        """
        Apply log transformation to all positive columns.
        """
        print("Applying log transformation...")
        for col in df.columns:
            if df[col].dtype in ["float64", "int64"] and (df[col] > 0).all():
                df[col] = np.log1p(df[col])  # log1p handles log(1 + x)
            else:
                print(
                    f"Skipping log transform for column '{col}' (non-numeric or non-positive values)."
                )
        return df

