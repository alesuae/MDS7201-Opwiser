import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from base_model import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self, max_depth=None):
        super().__init__()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
