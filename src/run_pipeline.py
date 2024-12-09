from src.data.data_preprocessing.prepare_data import DataPreparer
from src.data.data_preprocessing.data_splitter import DataSplitter
# TODO: HACERLOOOOOOOAAAAAA
from src.models.utils.model_config import get_config
from src.models.rf_model import RandomForestModel
from models.interpretability import explain_with_shap, plot_feature_importance
from sklearn.metrics import mean_squared_error
from src.data.main import data_pipeline

# Load data
X_train, X_val, X_test, y_train, y_val, y_test = data_pipeline()

# Models
model = RandomForestModel(max_depth=5)
model.train(X_train, y_train)
y_pred = model.predict(X_test)

# Interpretability
# SHAP
explain_with_shap(model.model, X_test)
# Feature Importance
plot_feature_importance(model.model, X_test.columns)

# Evaluate
rmse = model.evaluate(X_test, y_test, mean_squared_error)
print(f"RMSE: {rmse}")
