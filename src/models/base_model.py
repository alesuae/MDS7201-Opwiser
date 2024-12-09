from abc import ABC, abstractmethod
from sklearn import metrics

class BaseModel(ABC):
    def __init__(self):
        self.model = None

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

    def evaluate(self, X_test, y_test, metric_fn):
        y_pred = self.predict(X_test)
        return metric_fn(y_test, y_pred)
