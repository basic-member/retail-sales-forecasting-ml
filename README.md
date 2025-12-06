Retail Sales Forecasting (Scikit Learn)

This project builds a complete machine learning pipeline for predicting weekly retail sales.
Multiple datasets are merged, cleaned, processed, and used to train regression models with hyperparameter tuning.

Features

Merging train.csv, test.csv, features.csv, and stores.csv

Date feature extraction (Year, Month, Week)

Categorical encoding for holiday flags and store types

Missing value imputation

Train/test split

Model selection using GridSearchCV

Performance comparison using R²

Exporting final predictions for UI animation

Models Used

Linear Regression

Random Forest Regressor

The best model is automatically selected based on R² score.

Requirements
pandas
numpy
scikit-learn
matplotlib
wxPython (optional, for animation UI)

How to Run
python main.py
