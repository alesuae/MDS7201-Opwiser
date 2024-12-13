import itertools
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings

# Ignorar advertencias para optimización
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Cargar y_train desde el archivo CSV
y_train_path = "data/splits/y_train.csv"
y_train = pd.read_csv(y_train_path, index_col=0, parse_dates=True)

# Definir los rangos de parámetros para la búsqueda
p = d = q = range(0, 2)
P = D = Q = range(0, 2)
s = [52]  # Estacionalidad semanal

# Generar todas las combinaciones de parámetros
parameters = list(itertools.product(p, d, q, P, D, Q, s))

# Variable para guardar los mejores parámetros y el AIC más bajo
best_aic = float("inf")
best_params = None

print("Buscando los mejores hiperparámetros para SARIMAX...")

for param in parameters:
    try:
        # Ajustar el modelo SARIMAX
        order = param[:3]
        seasonal_order = param[3:]
        model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order)
        results = model.fit(disp=False)
        current_aic = results.aic

        # Imprimir el resultado
        print(f"SARIMAX{order}x{seasonal_order} - AIC:{current_aic}")

        # Actualizar los mejores parámetros si el AIC es menor
        if current_aic < best_aic:
            best_aic = current_aic
            best_params = (order, seasonal_order)

    except Exception as e:
        print(f"Error en combinación {param}: {e}")

# Imprimir los mejores parámetros encontrados
print("\nMejores parámetros encontrados:")
print(f"Order: {best_params[0]}, Seasonal Order: {best_params[1]}, AIC: {best_aic}")
