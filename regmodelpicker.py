import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class RegressionModelPicker:
    def __init__(self):
        self.X = None
        self.y = None
        self.mult_linear_obj = None

    def mult_linear(self, x_train, x_test,y_train, y_test):
        self.mult_linear_obj = LinearRegression()
        self.mult_linear_obj.fit(x_train, y_train)
        score = r2_score(y_test,self.mult_linear_obj.predict(x_test))
        return score
