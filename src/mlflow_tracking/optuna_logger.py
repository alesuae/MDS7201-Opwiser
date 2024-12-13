import optuna
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import json
import os

from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from optuna.visualization import plot_param_importances

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def optimize_model_with_optuna(
    model_class,
    param_distributions,
    X_train,
    y_train,
    X_test,
    y_test,
    n_trials,
    metric=mean_squared_error,
    greater_is_better=False
):
    """
    Optimiza un modelo de regresión usando Optuna y registra los resultados en MLFlow.

    Args:
        model_class: Clase del modelo (e.g., RandomForestRegressor, XGBRegressor).
        param_distributions (dict): Diccionario de distribuciones de parámetros a optimizar.
        X_train, y_train: Datos de entrenamiento.
        X_test, y_test: Datos de prueba.
        n_trials (int): Número de trials para Optuna.
        metric (callable): Métrica de evaluación para optimizar (por defecto: mean_squared_error).
        greater_is_better (bool): Si la métrica debe maximizarse o minimizarse.

    Returns:
        study: Objeto de estudio 
    """
    direction = "maximize" if greater_is_better else "minimize"

    def objective(trial):
        # Sugerir hiperparámetros
        params = {key: trial._suggest(key, value) for key, value in param_distributions.items()}

        # Crear instancia del modelo con los parámetros sugeridos
        model = model_class.set_params(**params, random_state=42)

        # Crear pipeline (si es necesario añadir preprocesadores)
        pipeline = Pipeline([
            ('regressor', model)
        ])

        # Validación cruzada para calcular la métrica
        scorer = make_scorer(metric, greater_is_better=greater_is_better)
        score = cross_val_score(pipeline, X_train, y_train, n_jobs=-1, cv=3, scoring=scorer).mean()

        return score

    # Crear un estudio Optuna para optimizar la métrica seleccionada
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)

    with mlflow.start_run(run_name=f"{str(model_class)} Optimization"):
        # Mejor trial
        best_trial = study.best_trial

        # Registrar hiperparámetros y métrica
        mlflow.log_params(best_trial.params)
        mlflow.log_metric("best_value", best_trial.value)

        # Registrar columnas usadas en el modelo
        with open("columns.json", "w") as f:
            json.dump(list(X_train.columns), f)
        mlflow.log_artifact("columns.json", artifact_path="model_metadata")

        # Gráfico de importancia de parámetros
        fig = plot_param_importances(study)
        artifact_name = f"param_importances_{type(model_class).__name__}.png"
        temp_file = f"temp_{artifact_name}"
        fig.write_image(temp_file)
        mlflow.log_artifact(temp_file, artifact_path="plots")
        os.remove(temp_file)

        print("Estudio de Optuna registrado en MLFlow.")

    return study
