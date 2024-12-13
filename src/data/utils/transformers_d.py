import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DateTransformer(BaseEstimator, TransformerMixin):
    """
    Transformador para convertir columnas de timestamp en componentes de fecha (año, mes, día).
    """
    def __init__(self):
        self.feature_names = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        transformed = pd.DataFrame()  # Crear un dataframe temporal
        for col in X.columns:
            transformed[f"{col}_year"] = pd.to_datetime(
                X[col], unit="s"
            ).dt.year.astype("Int64")
            transformed[f"{col}_month"] = pd.to_datetime(
                X[col], unit="s"
            ).dt.month.astype("Int64")
            transformed[f"{col}_day"] = pd.to_datetime(X[col], unit="s").dt.day.astype(
                "Int64"
            )

            self.feature_names.extend([
                f"{col}_year",
                f"{col}_month",
                f"{col}_day",
            ])
        return transformed.values
    
    def get_feature_names_out(self, input_features=None):
        return self.feature_names

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