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

    # Naming conventions from https://en.wikipedia.org/wiki/Polynomial_regression
    cached_coefficients_of_design_matrix = np.array([np.sum(x ** p) for p in range(degree * 2 + 1)])
    design_matrix = np.array([[cached_coefficients_of_design_matrix[i + j] for j in range(degree + 1)] for i in range(degree + 1)])
    response_vector = np.array([np.sum(y * (x ** i)) for i in range(degree + 1)])
    parameter_vector = np.linalg.solve(design_matrix, response_vector)

    # `axis=0` sums across polynomial terms, not data points
    # This avoids collapsing everything into a single scalar
    return lambda x_: np.sum([parameter_vector[i] * (x_ ** i) for i in range(degree + 1)], axis=0)
