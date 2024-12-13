import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
from src.mlflow_tracking.artifact_logger import log_plot_to_mlflow
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import json

def log_model_with_mlflow(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    run_name,
    params,
    register_model=False,
):
    """
    Registra el modelo, métricas y parámetros en MLFlow. Opcionalmente, registra el modelo en el Model Registry.

    Args:
        model: Modelo a registrar.
        X_train, y_train: Conjunto de entrenamiento.
        X_test, y_test: Conjunto de prueba.
        run_name (str): Nombre del run en MLFlow.
        params (dict): Hiperparámetros del modelo.
        register_model (bool): Si True, registra el modelo en el Model Registry.
    """
    with mlflow.start_run(run_name=run_name):
        start_time = time.time()
        # Entrenar el modelo
        model.fit(X_train, y_train)
        # Predicciones y métricas
        y_pred = model.predict(X_test)


        elapsed_time = time.time() - start_time
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        wape = np.mean(np.abs((np.array(y_test) - np.array(y_pred)) / np.array(y_test))) * 100

        print("Generando graficos de prediccion...")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Valores reales")
        ax.set_ylabel("Predicciones")
        ax.set_title("Predicciones vs Reales")
        ax.legend()
        artifact_path = "plots"
        temp_file = f"temp_predictions_{type(model).__name__}.png"
        fig.savefig(temp_file)
        mlflow.log_artifact(temp_file, artifact_path)
        plt.close(fig)
        os.remove(temp_file)

        residuals = [e1 - e2 for e1, e2 in zip(np.array(y_test),np.array(y_pred))]
        fig, ax = plt.subplots()
        ax.scatter(np.array(y_test), np.array(residuals), alpha=0.5)
        ax.axhline(0, color='red', linestyle='--')
        ax.set_xlabel("Valores reales")
        ax.set_ylabel("Residuos")
        ax.set_title("Residuos vs Valores Reales")
        ax.legend()
        artifact_path = "plots"
        temp_file = f"temp_residuals_{type(model).__name__}.png"
        fig.savefig(temp_file)
        mlflow.log_artifact(temp_file, artifact_path)
        plt.close(fig)
        os.remove(temp_file)

        fig, ax = plt.subplots()
        ax.hist(np.array(residuals), bins=30, alpha=0.7, color='blue')
        ax.set_xlabel("Residuos")
        ax.set_ylabel("Frecuencia")
        ax.set_title("Distribución de Residuos")
        ax.legend()
        artifact_path = "plots"
        temp_file = f"temp_distribucionResiduos_{type(model).__name__}.png"
        fig.savefig(temp_file)
        mlflow.log_artifact(temp_file, artifact_path)
        plt.close(fig)
        os.remove(temp_file)

        if hasattr(model, "feature_importances_"):
            feature_importance = model.feature_importances_
            sorted_idx = np.argsort(feature_importance)[::-1]
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.barh(np.array(X_train.columns)[sorted_idx][:12], feature_importance[sorted_idx][:12])
            ax.set_title("Importancia de Características")
            ax.set_xlabel("Importancia")
            ax.legend()
            artifact_path = "plots"
            temp_file = f"temp_featureImportance_{type(model).__name__}.png"
            fig.savefig(temp_file)
            mlflow.log_artifact(temp_file, artifact_path)
            plt.close(fig)
            os.remove(temp_file)

        metrics = {
            "elapsed_time": float(elapsed_time),
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "r2": float(r2),
            "wape": float(wape),
        }

        # Registrar parámetros y métricas
        mlflow.log_metrics(metrics)
        mlflow.log_params(params)
        signature = infer_signature(X_train, y_pred)

        # Registrar el modelo
        with open("columns.json", "w") as f:
            json.dump(list(X_train.columns), f)
        mlflow.log_artifact("columns.json", artifact_path="model_metadata")

        input_example = X_train.iloc[[0]]
        mlflow.sklearn.log_model(sk_model=model,
                                 artifact_path="model",
                                 input_example=input_example,
                                 registered_model_name=f"{type(model).__name__}_model",
                                 signature=signature,
                                 )

        print(f"Modelo registrado en MLFlow: {mlflow.active_run().info.run_id}")
        model_name = type(model).__name__

        # Registrar en el Model Registry si se especifica
        if register_model:
            if not model_name:
                raise ValueError(
                    "Debe proporcionar 'model_name' para registrar el modelo en el Model Registry."
                )
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            mlflow.register_model(model_uri, model_name)
            print(
                f"Modelo registrado en el Model Registry con el nombre '{model_name}'."
            )
