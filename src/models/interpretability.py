import shap
import matplotlib.pyplot as plt

# SHAP Values
def explain_with_shap_general(model, X):
    """
    Generate SHAP explanations for a given model and dataset.

    Args:
        model: Trained model.
        X (pd.DataFrame): Feature dataset.

    Returns:
        None (displays SHAP summary plot).
    """
    # Detecta el tipo de modelo y selecciona el explainer adecuado
    if hasattr(model, "tree_"):
        explainer = shap.TreeExplainer(model)  # Modelos basados en árboles
    elif hasattr(model, "coef_"):
        explainer = shap.LinearExplainer(model, X)  # Modelos lineales
    else:
        explainer = shap.KernelExplainer(model.predict, X)  # Otros modelos

    # Calcula los SHAP values
    shap_values = explainer.shap_values(X)

    # Gráficos SHAP
    shap.summary_plot(shap_values, X)


def explain_with_shap(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Summary plot
    shap.summary_plot(shap_values, X_test)

    # Dependence plot for a specific variable
    shap.dependence_plot("nombre_variable", shap_values, X_test)

    #choosen_instance = X_test.loc[[421]]
    #shap_values = explainer.shap_values(choosen_instance)
    #shap.initjs()
    #shap.force_plot(explainer.expected_value[1], shap_values[1], choosen_instance) 

# Feature importance
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    plt.bar(feature_names, importances)
    plt.title("Feature Importance")
    plt.show()

# Partial dependence plot

