import pandas as pd
import matplotlib.pyplot as plt
from src.models.arima_model import ARIMAModel
from src.data.main import data_pipeline

#import sys
#import os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Ejecutar el pipeline para obtener los conjuntos de datos
print("Ejecutando el pipeline para cargar los datos...")
X_train, X_val, X_test, y_train, y_val, y_test = data_pipeline()

# Visualizar y verificar los datos
print("Primeros valores de y_train:")
print(y_train.head())

# Confirmar que y_train tiene un índice de fechas
if not isinstance(y_train.index, pd.DatetimeIndex):
    print("El índice de y_train no es un DatetimeIndex. Convirtiendo...")
    y_train.index = pd.to_datetime(y_train.index)

# Crear e inicializar el modelo ARIMA
arima_model = ARIMAModel(order=(1, 1, 1))

# Entrenar el modelo con y_train
print("Entrenando el modelo ARIMA con datos reales...")
arima_model.fit(y_train)

# Generar predicciones para los datos de prueba
steps = len(y_test)  # Predecir el mismo número de pasos que y_test
predictions = arima_model.predict(steps=steps)

# Guardar predicciones como CSV
predictions.to_csv('src/results/arima_model/metrics/arima_predictions_real.csv', header=['Predicciones'])
print("Predicciones guardadas en 'src/results/arima_model/metrics/arima_predictions_real.csv'.")

# Graficar predicciones vs. valores reales
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Datos reales')
plt.plot(predictions, label='Predicciones', linestyle='--')
plt.legend()
plt.title("Predicciones ARIMA vs. Datos Reales")
plt.savefig('src/results/arima_model/plots/arima_real_vs_pred.png')
plt.show()
print("Gráfico guardado en 'src/results/arima_model/plots/arima_real_vs_pred.png'.")

# Guardar resumen del modelo en logs/
arima_model.save_results('src/results/arima_model/logs/arima_summary_real.txt')
print("Resumen del modelo guardado en 'src/results/arima_model/logs/arima_summary_real.txt'.")
