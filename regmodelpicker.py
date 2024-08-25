import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score
from sklearn.svm import SVR


class RegressionModelPicker:
    def __init__(self):
        self.X = None
        self.y = None
        self.mult_linear_reg = None
        self.poly_reg = None
        self.poly_feat = None
        self.svr_reg = None

    def linear_model(self, x_train, x_test, y_train, y_test):
        self.mult_linear_reg = LinearRegression()
        self.mult_linear_reg.fit(x_train, y_train)
        score = r2_score(y_test, self.mult_linear_reg.predict(x_test))
        return score

    def polynomial_model(self, degree, x_train, x_test, y_train, y_test):
        self.poly_feat = PolynomialFeatures(degree=degree)
        X_poly = self.poly_feat.fit_transform(x_train)
        self.poly_reg = LinearRegression()
        self.poly_reg.fit(X_poly, y_train)
        score = r2_score(y_test, self.poly_reg.predict(self.poly_feat.transform(x_test)))
        return score

    def svr_model(self, x_train, x_test, y_train, y_test, feat_scale):
        self.svr_reg = SVR(kernel="rbf")
        if feat_scale:
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
            x_train = x_train.reshape(-1, 1)
            y_train = y_train.reshape(-1, 1)
            x_train = scaler_x.fit_transform(x_train)
            y_train = scaler_y.fit_transform(y_train)
            self.svr_reg.fit(x_train,y_train)
            y_pred = self.svr_reg.predict(scaler_x.transform(x_test))
            score = r2_score(y_test,scaler_y.inverse_transform(y_pred))
            return score
        else:
            self.svr_reg.fit(x_train,y_train)
            y_pred = self.svr_reg.predict(x_test)
            score = r2_score(y_test,y_pred)
            return score


