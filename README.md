# Regression Model Picker

A Python utility class that runs five regression models on any dataset and compares their R² scores in one call.

## What It Does

Pass in your train/test data and it runs all five regressors, returns the trained model and R² score for each. Or call `best_reg_model()` to run all five at once and compare performance.

## Models Included

- Linear Regression
- Polynomial Regression (configurable degree)
- Support Vector Regression (SVR)
- Decision Tree Regression
- Random Forest Regression

## Usage

```python
from regression_model_picker import RegressionModelPicker

picker = RegressionModelPicker()

# run all models at once
picker.best_reg_model(X_train, X_test, y_train, y_test)

# or run individually
model, score = picker.random_forest_model(X_train, X_test, y_train, y_test, n_trees=100)
```

## Parameters

Each model has sensible defaults but you can customise:
- `polynomial_model(degree=4)`
- `svr_model(feat_scale=True)` - automatically scales features for SVR
- `random_forest_model(n_trees=10)`

## Features

- SVR model handles feature scaling automatically when `feat_scale=True`
- Polynomial regression uses `PolynomialFeatures` to transform inputs before fitting
- All models evaluated using R² score on the test set

## Requirements
```
pip install numpy scikit-learn
```
