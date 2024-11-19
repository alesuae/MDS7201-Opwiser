from prophet import Prophet
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import itertools
from tqdm import tqdm
import shap
import pickle
import matplotlib.pyplot as plt

class PROPHETModel:
    def __init__(self, seasonality_mode='additive', yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, changepoint_prior_scale=0.05):
        """
        Inicializa el modelo Prophet con los parámetros básicos.
        seasonality_mode: 'additive' o 'multiplicative'.
        yearly_seasonality: incluir estacionalidad anual.
        weekly_seasonality: incluir estacionalidad semanal.
        daily_seasonality: incluir estacionalidad diaria.
        changepoint_prior_scale: controla la flexibilidad de los puntos de cambio.
        """
        self.model = None
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale

    def preprocess_data(self, X, y):
        """
            Preprocesa los datos: renombra las columnas y convierte 'fecha' a datetime.
            X: DataFrame con las variables exógenas.
            y: Series o DataFrame con la variable objetivo.
            """
        if 'fecha' not in X:
            raise ValueError("El DataFrame X debe contener una columna llamada 'fecha'.")
        
        if len(X) != len(y):
            raise ValueError("El tamaño de X y y no coincide.")
        
        # Cambiar nombres de las columnas
        X = X.copy()  # Asegurar que no se modifique el DataFrame original
        X['ds'] = pd.to_datetime(X['fecha'], errors='coerce')  # Asegura conversión válida
        if X['ds'].isnull().any():
            raise ValueError("Algunas fechas no pudieron convertirse a formato datetime.")
        
        X['y'] = y.values if isinstance(y, pd.Series) else y  # Asegurar que y es una Serie/array
        return X[['ds', 'y'] + [col for col in X.columns if col not in ['fecha', 'y', 'ds']]]

    def fit(self, X_train, y_train):
        """Entrena el modelo Prophet con los datos de entrenamiento."""
        df_train = self.preprocess_data(X_train, y_train)
        self.model = Prophet(
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale
        )
        self.model.fit(df_train)

    def predict(self, X_test):
        """Genera predicciones a futuro."""
        # Copiar los datos para no modificar el original
        X_test = X_test.copy()

        # Verificar y procesar la columna 'fecha' si existe
        if 'fecha' in X_test.columns:
            X_test['ds'] = pd.to_datetime(X_test['fecha'], errors='coerce')
            if X_test['ds'].isnull().any():
                raise ValueError("Algunas fechas en la columna 'fecha' no pudieron convertirse a formato datetime.")
        elif 'ds' in X_test.columns:
            X_test['ds'] = pd.to_datetime(X_test['ds'], errors='coerce')
            if X_test['ds'].isnull().any():
                raise ValueError("Algunas fechas en la columna 'ds' no pudieron convertirse a formato datetime.")
        else:
            raise ValueError("El DataFrame X_test debe contener una columna 'fecha' o 'ds'.")

        # Filtrar solo las columnas necesarias
        X_test = X_test[['ds'] + [col for col in X_test.columns if col not in ['fecha', 'ds']]]

        # Generar predicciones
        forecast = self.model.predict(X_test)
        return forecast  # [['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    def save_model(self, path):
        """
        Guarda el modelo Prophet en un archivo.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, path):
        """
        Carga un modelo Prophet desde un archivo.
        """
        with open(path, 'rb') as f:
            self.model = pickle.load(f)

    def plot(self, forecast, title=None, save_path=None, figsize=(10, 8)):
        """
        Grafica las predicciones generadas por el modelo Prophet con un título opcional.
        forecast: DataFrame de predicciones generado por Prophet.
        title: Título opcional para el gráfico.
        save_path: Ruta opcional para guardar el gráfico.
        figsize: Tupla opcional para especificar el tamaño de la figura (ancho, alto).
        """
        if self.model:
            fig = self.model.plot(forecast)
            fig.set_size_inches(*figsize)

            if title:
                plt.title(title, fontsize=16)
            
            if save_path:
                fig.savefig(save_path, bbox_inches='tight')

            # plt.show()

    def plot_components(self, forecast, title=None, save_path=None, figsize=(10, 12)):
        """
        Grafica los componentes del modelo Prophet con un título opcional.
        forecast: DataFrame de predicciones generado por Prophet.
        title: Título opcional para el gráfico.
        save_path: Ruta opcional para guardar el gráfico.
        figsize: Tupla opcional para especificar el tamaño de la figura (ancho, alto).
        """
        if self.model:  
            fig = self.model.plot_components(forecast)
            fig.set_size_inches(*figsize)

            if title:
                plt.gcf().suptitle(title, fontsize=16)
            
            # Ajusta los márgenes para que no se superpongan títulos y componentes
            # plt.tight_layout(rect=[0, 0, 1, 0.96])  

            if save_path:
                fig.savefig(save_path, bbox_inches='tight')

            # plt.show()

    def optimize_hyperparameters(self, X_train, X_val, y_train, y_val, param_grid, features=[]):
        """
        Optimiza los hiperparámetros de Prophet utilizando una función de optimización.
        
        X_train, X_val, X_test: DataFrames con las variables exógenas (características).
        y_train, y_val, y_test: Series o DataFrames con las variables objetivo.
        param_grid: Diccionario con los rangos de parámetros a optimizar.
        features: Lista de características adicionales para añadir como regresores (por defecto es vacío).
        """
        # Preprocesamiento de datos
        df_train = self.preprocess_data(X_train, y_train)
        df_val = self.preprocess_data(X_val, y_val)
        
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        MAEs = []

        # Si no se proporcionan features, se asegura que features esté vacío
        if not features:
            features = []

        for params in tqdm(all_params, desc="Tuning Prophet parameters"):
            m = Prophet(**params)
            
            # Añadir las características (features) como regresores si existen
            for feature in features:
                m.add_regressor(feature)
            
            m.fit(df_train)

            # Prepare future dataframe
            df_prop = m.make_future_dataframe(periods=len(df_val))
            df_feat = pd.concat([df_train[features], df_val[features]]).reset_index(drop=True)
            df_prop[features] = df_feat[features]

            # Hacer las predicciones
            forecast = m.predict(df_prop)
            forecast_pred = forecast[forecast['ds'].isin(df_val['ds'])].reset_index(drop=True)

            print("Longitud de df_val:", len(df_val))
            print("Longitud de forecast_pred:", len(forecast_pred))

            # Calcular MAE
            mae_pred = mean_absolute_error(df_val['y'], forecast_pred['yhat'])
            MAEs.append(mae_pred)

        # Ordenar los resultados por el MAE
        tuning_results = pd.DataFrame(all_params)
        tuning_results['MAEs'] = MAEs
        tuning_results = tuning_results.sort_values(by='MAEs', ascending=True)
        
        # Obtener los mejores parámetros
        best_params = all_params[np.argmin(MAEs)]
        
        return tuning_results, best_params

    def calculate_mae_rmse(self, y_true, y_pred):
        """Calcula MAE y RMSE."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return mae, rmse

    def add_regressors(self, df, features):
        """Agrega regresores exógenos al modelo."""
        for feature in features:
            self.model.add_regressor(feature)

    def calculate_shap_values(self, df, model=None):
        """Calcula los SHAP values para interpretar el modelo."""
        if not model:
            model = self.model
        explainer = shap.ProphetExplainer(model)
        shap_values = explainer.shap_values(df)
        return shap_values

    def plot_shap_summary(self, shap_values):
        """Genera un gráfico de resumen de los SHAP values."""
        shap.summary_plot(shap_values)
