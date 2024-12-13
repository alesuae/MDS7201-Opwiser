import optuna
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

def get_param_distributions(model, seed=42):
    """
    Devuelve las distribuciones de hiperparámetros para el modelo dado.

    Args:
        model: Modelo del cual se quieren optimizar los hiperparámetros.
        seed (int): Semilla para garantizar reproducibilidad.

    Returns:
        dict: Diccionario de distribuciones de parámetros para Optuna.
    """
    if isinstance(model, RandomForestRegressor):
        return {
            "n_estimators": optuna.distributions.IntDistribution(50, 300),
            "max_depth": optuna.distributions.IntDistribution(3, 20),
            "min_samples_split": optuna.distributions.IntDistribution(2, 20),
            "min_samples_leaf": optuna.distributions.IntDistribution(1, 10),
            "max_features": optuna.distributions.CategoricalDistribution(["sqrt", "log2", None]),
        }
    elif isinstance(model, LGBMRegressor):
        return {
            "num_leaves": optuna.distributions.IntDistribution(10, 100),
            "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.3),
            "n_estimators": optuna.distributions.IntDistribution(50, 300),
            "max_depth": optuna.distributions.IntDistribution(3, 6),
            "subsample": optuna.distributions.FloatDistribution(0.5, 1.0),
            "min_gain_to_split": optuna.distributions.FloatDistribution(0.1, 0.5),
            "min_data_in_leaf": optuna.distributions.IntDistribution(5, 10, 20),
        }
    elif isinstance(model, XGBRegressor):
        return {
            "n_estimators": optuna.distributions.IntDistribution(50, 300),
            "max_depth": optuna.distributions.IntDistribution(3, 20),
            "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.3),
            "subsample": optuna.distributions.FloatDistribution(0.5, 1.0),
            "colsample_bytree": optuna.distributions.FloatDistribution(0.5, 1.0),
            "reg_alpha": optuna.distributions.FloatDistribution(0.0, 10.0),
            "reg_lambda": optuna.distributions.FloatDistribution(0.0, 10.0),
        }
    elif isinstance(model, DecisionTreeRegressor):
        return {
            "max_depth": optuna.distributions.IntDistribution(3, 20),
            "min_samples_split": optuna.distributions.IntDistribution(2, 20),
            "min_samples_leaf": optuna.distributions.IntDistribution(1, 10),
        }
    elif isinstance(model, LinearRegression):
        return {
            "fit_intercept": optuna.distributions.CategoricalDistribution([True, False]),
            "normalize": optuna.distributions.CategoricalDistribution([True, False]),
        }
    else:
        raise ValueError(f"No se han definido distribuciones de parámetros para el modelo: {type(model)}")
