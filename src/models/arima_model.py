import os
import json
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX 
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

class SARIMAXModel:
    def __init__(self, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0), result_dir="src/results/arima_model"):
        """
        Inicializa el modelo SARIMAX con órdenes específicos y define el directorio para guardar resultados.
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        self.result_dir = result_dir

        # Crear carpetas si no existen
        os.makedirs(os.path.join(self.result_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.result_dir, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(self.result_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.result_dir, "plots"), exist_ok=True)

    def fit(self, y, exog=None):
        """
        Ajusta el modelo SARIMAX a los datos de entrenamiento y guarda el resumen en un archivo de logs.
        """
        self.model = SARIMAX(y, order=self.order, seasonal_order=self.seasonal_order, exog=exog)
        self.fitted_model = self.model.fit(disp=False)

        # Guardar resumen del modelo
        log_path = os.path.join(self.result_dir, "logs", f"fit_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(log_path, "w") as f:
            f.write(self.fitted_model.summary().as_text())
        print(f"Resumen del modelo guardado en: {log_path}")

    def predict(self, steps=10, exog=None):
        """
        Genera predicciones a futuro.
        """
        if not self.fitted_model:
            raise ValueError("El modelo debe ser ajustado antes de predecir.")
        forecast = self.fitted_model.forecast(steps=steps, exog=exog)
        return forecast

    def evaluate(self, y_true, y_pred):
        """
        Calcula las métricas MAE y RMSE y las guarda en un archivo JSON.
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        return mae, rmse

    def save_metrics(self, mae, rmse, save_path):
        """
        Guarda las métricas MAE y RMSE en un archivo JSON.
        """
        metrics = {
            "mae": mae,
            "rmse": rmse,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Métricas guardadas en: {save_path}")

    def save_plot(self, y_train, forecast, steps, file_path):
        """
        Guarda un gráfico de la serie temporal y las predicciones, limitado a 52 semanas.
        """
        plt.figure(figsize=(10, 6))
        # Plotear los datos de entrenamiento
        plt.plot(y_train, label="Datos de entrenamiento")
    
        # Limitar el rango de predicción a 52 semanas
        steps = min(steps, 52)
        future_index = pd.date_range(y_train.index[-1], periods=steps + 1, freq="W")[1:]
    
        # Asegurar que forecast tiene la misma longitud que future_index
        plt.plot(future_index, forecast[:steps], label="Predicción", color='orange')
        plt.legend()
        plt.title("SARIMAX Predicción (Limitado a 52 semanas)")
        plt.savefig(file_path, format='png')
        plt.close()  # Cierra el gráfico para evitar mostrarlo en pantalla
        print(f"Gráfico guardado en: {file_path}")

    def save_model(self, file_path):
        """
        Guarda el modelo ajustado en un archivo pickle.
        """
        if not self.fitted_model:
            raise ValueError("El modelo debe ser ajustado antes de guardarlo.")
        with open(file_path, 'wb') as file:
            pickle.dump(self.fitted_model, file)
        print(f"Modelo guardado en: {file_path}")
