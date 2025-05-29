from typing import Callable
import numpy as np
import pandas as pd


type RegressionModel = Callable[[float], float]


def apply_regression_model(df: pd.DataFrame, x_column: str, y_column: str, model: RegressionModel) -> pd.DataFrame:
    df[y_column] = model(df[x_column])
    return df

def train_polynomial_regression_model(df: pd.DataFrame, x_column: str, y_column: str, degree: int) -> RegressionModel:
    return __train_polynomial_regression_model(x=df[x_column], y=df[y_column], degree=degree)

def __train_polynomial_regression_model(x: np.ndarray, y: np.ndarray, degree: int):
    assert len(x) == len(y)

    # https://en.wikipedia.org/wiki/Polynomial_regression
    X = np.vander(x, N=degree + 1, increasing=True)
    β = np.linalg.inv(X.T @ X) @ X.T @ y

    return lambda x: np.vander(x, N=degree + 1, increasing=True) @ β

def train_ridge_regression_model(df: pd.DataFrame, x_column: str, y_column: str, degree: int, λ: float) -> RegressionModel:
    return __train_ridge_regression_model(x=df[x_column], y=df[y_column], degree=degree, λ=λ)

def __train_ridge_regression_model(x: np.ndarray, y: np.ndarray, degree: int, λ: float):
    assert len(x) == len(y)
    
    # https://en.wikipedia.org/wiki/Ridge_regression
    X = np.vander(x, N=degree + 1, increasing=True)
    I = np.eye(degree + 1)
    β = np.linalg.inv(X.T @ X + λ * I) @ X.T @ y

    return lambda x: np.vander(x, N=degree + 1, increasing=True) @ β

def train_lasso_regression_model(df: pd.DataFrame, x_column: str, y_column: str, degree: int, λ: float) -> RegressionModel:
    return __train_lasso_regression_model(x=df[x_column], y=df[y_column], degree=degree, λ=λ)

def __train_lasso_regression_model(x: np.ndarray, y: np.ndarray, degree: int):
    assert len(x) == len(y)
