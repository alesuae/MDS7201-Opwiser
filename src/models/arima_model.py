from statsmodels.tsa.arima.model import ARIMA

class ARIMAModel:
    def __init__(self, order=(1, 1, 1)):
        """
        Inicializa el modelo ARIMA con los parámetros de orden.
        order: (p, d, q)
            - p: número de términos autoregresivos (AR)
            - d: número de diferenciaciones necesarias para estacionariedad (I)
            - q: número de términos de promedio móvil (MA)
        """
        self.order = order
        self.model = None

    def fit(self, train_data):
        """Entrena el modelo ARIMA con los datos de entrenamiento."""
        self.model = ARIMA(train_data, order=self.order).fit()

    def predict(self, steps):
        """Genera predicciones a futuro."""
        return self.model.forecast(steps=steps)

    def save_results(self, path):
        """Guarda los resultados del modelo en un archivo."""
        with open(path, 'w') as f:
            f.write(str(self.model.summary()))
