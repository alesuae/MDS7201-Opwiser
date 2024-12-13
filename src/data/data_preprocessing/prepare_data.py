import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    PowerTransformer,
    OneHotEncoder,
    LabelEncoder,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.data.utils.config import get_config
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

    def prepare(self, df):
        """
        Prepare the dataset according to the current configuration.
        """
        original_columns = df.columns
        df = self.impute_missing_values(df)

        # Apply log transform if enabled
        df = self.log_transform(df)

        timestamp_columns = df.select_dtypes(include=["datetime64"]).columns
        numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns
        df_ns = df.copy()
        df_ns = df_ns.drop(columns=['semana'])
        categorical_columns = df_ns.select_dtypes(include=["object", "category", "string"]).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ("date", SemanaTransformer(), ['semana']),
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
        num_column_names = preprocessor.named_transformers_["num"].steps[0][1].get_feature_names_out()
        cat_column_names = preprocessor.named_transformers_["cat"].get_feature_names_out()

        transformed_column_names = np.append(
            date_column_names, num_column_names
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


class SemanaTransformer(BaseEstimator, TransformerMixin):
    """
    Transformador para procesar rangos de fechas en la columna 'semana'.
    Extrae características útiles para forecasting.
    """

    def __init__(self):
        self.feature_names = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        transformed = pd.DataFrame()

        for col in X.columns:
            # Dividir el rango en fecha de inicio y fin
            transformed[f"{col}_start"] = X[col].str.split('/').str[0]
            transformed[f"{col}_end"] = X[col].str.split('/').str[1]

            # Convertir a formato datetime
            transformed[f"{col}_start"] = pd.to_datetime(transformed[f"{col}_start"])
            transformed[f"{col}_end"] = pd.to_datetime(transformed[f"{col}_end"])

            # Extraer características de la fecha de inicio
            transformed[f"{col}_start_year"] = transformed[f"{col}_start"].dt.year
            transformed[f"{col}_start_month"] = transformed[f"{col}_start"].dt.month
            transformed[f"{col}_start_weekday"] = transformed[f"{col}_start"].dt.weekday

            # Extraer características de la fecha de fin
            transformed[f"{col}_end_year"] = transformed[f"{col}_end"].dt.year
            transformed[f"{col}_end_month"] = transformed[f"{col}_end"].dt.month
            transformed[f"{col}_end_weekday"] = transformed[f"{col}_end"].dt.weekday

            # Opcional: calcular la duración del rango en días
            transformed[f"{col}_duration_days"] = (
                transformed[f"{col}_end"] - transformed[f"{col}_start"]
            ).dt.days

            # Generar nombres de columnas
            self.feature_names.extend([
                f"{col}_start_year",
                f"{col}_start_month",
                f"{col}_start_weekday",
                f"{col}_end_year",
                f"{col}_end_month",
                f"{col}_end_weekday",
                f"{col}_duration_days"
            ])

        # Eliminar columnas originales (si es necesario)
        transformed.drop(columns=[f"{col}_start", f"{col}_end"], inplace=True)

        return transformed.values

    def get_feature_names_out(self, input_features=None):
        return self.feature_names
