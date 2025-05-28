from typing import Callable
import numpy as np
import pandas as pd


type PolynomialRegressionModel = Callable[[float], float]


def apply_polynomial_regression_model(df: pd.DataFrame, x_column: str, y_column: str, model: PolynomialRegressionModel) -> pd.DataFrame:
    df[y_column] = model(df[x_column])
    return df

# TODO Impl Gaussian noise
def train_polynomial_regression_model(df: pd.DataFrame, x_column: str, y_column: str, degree: int) -> PolynomialRegressionModel:
    return __train_polynomial_regression_model(x=df[x_column], y=df[y_column], degree=degree)

def __train_polynomial_regression_model(x: np.ndarray, y: np.ndarray, degree: int):
    assert len(x) == len(y)

    # https://en.wikipedia.org/wiki/Polynomial_regression
    X = np.vander(x, N=degree + 1, increasing=True)
    β = np.linalg.inv(X.T @ X) @ X.T @ y

    return lambda x: np.vander(x, N=degree + 1, increasing=True) @ β
