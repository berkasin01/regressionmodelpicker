from regmodelpicker import RegressionModelPicker
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv("example_data.csv")

#data preprocess
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=14)

reg_obj = RegressionModelPicker()

reg_obj.best_reg_model(X_train, X_test, y_train, y_test)

linear_model = reg_obj.linear_model(X_train, X_test, y_train, y_test)[0]

poly_feature_scaler = reg_obj.polynomial_model(X_train, X_test, y_train, y_test,4)[0]
poly_model = reg_obj.polynomial_model(X_train, X_test, y_train, y_test,4)[1]

svr_model = reg_obj.svr_model(X_train, X_test, y_train, y_test, True)[0]

dt_model = reg_obj.decision_tree_model(X_train, X_test, y_train, y_test)[0]

rf_model = reg_obj.random_forest_model(X_train, X_test, y_train, y_test,10)[0]