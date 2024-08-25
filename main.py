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

print(reg_obj.linear_model(X_train, X_test, y_train, y_test))
print(reg_obj.polynomial_model(4, X_train, X_test, y_train, y_test))
print(reg_obj.svr_model(X_train, X_test, y_train, y_test,True))