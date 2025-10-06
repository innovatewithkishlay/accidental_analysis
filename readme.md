# Accident Risk Prediction

This project is a *machine learning regression pipeline* to predict accident risk based on a dataset of numerical and categorical features. It covers the full workflow from data preprocessing to model evaluation and submission generation.  

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dependencies](#dependencies)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Cross-Validation and Final Prediction](#cross-validation-and-final-prediction)
- [Usage](#usage)
- [Output](#output)

---

## Overview

The goal of this project is to predict the *accident_risk* for given samples using regression models. Multiple regression models are trained and evaluated, including:

- Linear Models: LinearRegression, Lasso, Ridge, ElasticNet
- Tree-Based Models: DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
- Advanced Boosting Models: XGBoost, CatBoost, LightGBM
- K-Nearest Neighbors: KNeighborsRegressor

A final model (XGBRegressor) is trained with *K-Fold cross-validation* and used to generate predictions for the test dataset.

---

## Features

- Detection of numerical and categorical columns
- Outlier visualization using boxplots
- Encoding categorical variables with OrdinalEncoder
- Feature scaling for numerical columns using StandardScaler
- Training multiple regression models and comparing performance
- K-Fold cross-validation for robust prediction
- Submission-ready CSV output

---

## Dependencies

The project uses the following Python libraries:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- xgboost
- catboost
- lightgbm
- torch (for checking GPU availability)
- warnings (to suppress warnings)

Install dependencies using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost catboost lightgbm torch