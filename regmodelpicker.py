import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


class RegressionModelPicker:
    def __init__(self):
        self.X = None
        self.y = None
        self.mult_linear_reg = None
        self.poly_reg = None
        self.poly_feat = None
        self.svr_reg = None
        self.dt_model = None
        self.forest_model = None

    def linear_model(self, x_train, x_test, y_train, y_test):
        self.mult_linear_reg = LinearRegression()
        self.mult_linear_reg.fit(x_train, y_train)
        score = r2_score(y_test, self.mult_linear_reg.predict(x_test))
        return self.mult_linear_reg, score

    def polynomial_model(self, x_train, x_test, y_train, y_test, degree=4):
        self.poly_feat = PolynomialFeatures(degree=degree)
        X_poly = self.poly_feat.fit_transform(x_train)
        self.poly_reg = LinearRegression()
        self.poly_reg.fit(X_poly, y_train)
        score = r2_score(y_test, self.poly_reg.predict(self.poly_feat.transform(x_test)))
        return self.poly_feat, self.poly_reg, score

    def svr_model(self, x_train, x_test, y_train, y_test, feat_scale=True):
        self.svr_reg = SVR(kernel="rbf")
        if feat_scale:
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()

            x_train = scaler_x.fit_transform(x_train)
            y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))

            self.svr_reg.fit(x_train, np.ravel(y_train))
            y_pred = self.svr_reg.predict(scaler_x.transform(x_test))
            score = r2_score(y_test, scaler_y.inverse_transform(y_pred.reshape(-1, 1)))
            return self.svr_reg, score
        else:
            self.svr_reg.fit(x_train, y_train)
            y_pred = self.svr_reg.predict(x_test)
            score = r2_score(y_test, y_pred)
            return self.svr_reg, score

    def decision_tree_model(self, x_train, x_test, y_train, y_test):
        self.dt_model = DecisionTreeRegressor()
        self.dt_model.fit(x_train, y_train)
        y_pred = self.dt_model.predict(x_test)
        score = r2_score(y_test, y_pred)
        return self.dt_model, score

    def random_forest_model(self, x_train, x_test, y_train, y_test, n_trees=10):
        self.forest_model = RandomForestRegressor(n_estimators=n_trees)
        self.forest_model.fit(x_train, y_train)
        y_pred = self.forest_model.predict(x_test)
        score = r2_score(y_test, y_pred)
        return self.forest_model, score

    def best_reg_model(self, x_train, x_test, y_train, y_test):

        linear_score = self.linear_model(x_train, x_test, y_train, y_test)[1]
        poly_score = self.polynomial_model(x_train, x_test, y_train, y_test, 4)[2]
        svr_score = self.svr_model(x_train, x_test, y_train, y_test, True)[1]
        dt_score = self.decision_tree_model(x_train, x_test, y_train, y_test)[1]
        rf_score = self.random_forest_model(x_train, x_test, y_train, y_test, 10)[1]

        dic = {"Linear Regression Score": linear_score,
               "Polynomial Regression Score": poly_score,
               "SVR Score": svr_score,
               "Decision Tree Regression Score": dt_score,
               "Random Forest Regression Score": rf_score
               }

        print(dic)



