from sklearn.model_selection import GridSearchCV

class ModelSelector:
    def __init__(self, model_config):
        self.model_config = model_config

    def run_grid_search(self, model, param_grid, X_train, y_train):
        """
        Ejecuta búsqueda de hiperparámetros usando validación cruzada.
        """
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=TimeSeriesSplit(n_splits=self.model_config['cv_splits']),
            scoring='neg_mean_squared_error',
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        return grid_search.best_params_, grid_search.best_score_

    def evaluate_model(self, model, X_train, X_val, y_train, y_val):
        """
        Entrena y evalúa un modelo con los mejores hiperparámetros.
        """
        model.fit(X_train, y_train)
        val_score = model.score(X_val, y_val)
        return val_score
