import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    PowerTransformer,
    OneHotEncoder,
    LabelEncoder,
)

from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import LabelEncoder
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

        timestamp_columns = ['fecha']
        date_column_original = df['fecha']

        if self.temporal_data:
            df = df.drop(columns=timestamp_columns)
            timestamp_columns = df.select_dtypes(include=["datetime64"]).columns

        categoria_column = df['categoria_2']
        semana_column = df['semana']
        df = df.drop(columns=['semana', 'categoria_2'])

        numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns
        categorical_columns = df.select_dtypes(include=["object", "category", "string"]).columns

        # Codificar manualmente las columnas categóricas con LabelEncoder
        label_encoders = {}
        encoded_columns = []

        # Aplicar LabelEncoder a cada columna categórica
        for col in categorical_columns:
            label_encoder = LabelEncoder()
            df[col] = label_encoder.fit_transform(df[col])
            label_encoders[col] = label_encoder  # Guardar el encoder para transformaciones inversas
            encoded_columns.append(col)

        preprocessor = ColumnTransformer(
            transformers=[
                ("date", DateTransformer(), timestamp_columns),
                ("encoded", "passthrough", encoded_columns),
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
                #(
                #    "cat",
                #    OneHotEncoder(),
                #    categorical_columns,
                #),
            ],
            remainder="passthrough",  # Dejar columnas no especificadas
        ).fit(df)

        date_column_names = np.array(preprocessor.named_transformers_["date"].get_feature_names_out())
        num_column_names = np.array(preprocessor.named_transformers_["num"].steps[0][1].get_feature_names_out())
        encoded_column_names = np.array(preprocessor.named_transformers_["encoded"].get_feature_names_out())

        transformed_column_names = np.append(
            date_column_names, num_column_names
        )
        transformed_column_names = np.append(
            transformed_column_names, encoded_column_names
        )
        preprocessor_pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
        processed_data = preprocessor_pipeline.fit_transform(df)

        processed_data_df = pd.DataFrame(
            processed_data, columns=transformed_column_names
        )

        # smote rc
        processed_data_df['categoria_2'] = categoria_column
        processed_data_df = self.apply_smote(processed_data_df, categorical_columns)

        processed_data_df['semana'] = semana_column
        processed_data_df['fecha'] = date_column_original

        return processed_data_df
    
    def apply_smote(self, df, categorical_columns):
        """
        Apply SMOTENC to balance the dataset.
        """
        # Identificar índices de las columnas categóricas
        categorical_indices = [df.columns.get_loc(col) for col in categorical_columns]

        # Separar X e y
        X = df.drop(columns=['categoria_2'])
        y = pd.DataFrame({'categoria_2':df['categoria_2']})

        # Codificar categorías con LabelEncoder
        label_encoder = LabelEncoder()
        y_encoded = np.array(label_encoder.fit_transform(y))

        # Calcular el número máximo de vecinos permitido
        from collections import Counter
        min_samples_per_class = y['categoria_2'].value_counts().min()
        n_neighbors = min(5, int(min_samples_per_class) - 1)  # Ajusta a la clase más pequeña

        # Aplicar SMOTENC
        smote_n = SMOTENC(categorical_features=categorical_indices,random_state=42, k_neighbors=n_neighbors)
        X_resampled, y_resampled = smote_n.fit_resample(X, y_encoded)

        # Reconstruir el DataFrame balanceado
        resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
        resampled_df['categoria_2'] = label_encoder.inverse_transform(y_resampled)

        return resampled_df

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

    