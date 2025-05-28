from typing import Optional
import numpy as np
import pandas as pd


# TODO Impl Gaussian noise
def fit_polynomial_regression(df: pd.DataFrame, x_column: str, y_column: str, degree: int, new_y_column: Optional[str] = None) -> pd.DataFrame:
    new_y_column = new_y_column or y_column
    df[new_y_column] = __fit_polynomial_regression(x=df[x_column], y=df[y_column], degree=degree)

def __fit_polynomial_regression(x: np.ndarray, y: np.ndarray, degree: int):
    assert len(x) == len(y)

    # Naming conventions from https://en.wikipedia.org/wiki/Polynomial_regression
    cached_coefficients_of_design_matrix = np.array([np.sum(x ** p) for p in range(degree * 2 + 1)])
    design_matrix = np.array([[cached_coefficients_of_design_matrix[i + j] for j in range(degree + 1)] for i in range(degree + 1)])
    response_vector = np.array([np.sum(y * (x ** i)) for i in range(degree + 1)])
    parameter_vector = np.linalg.solve(design_matrix, response_vector)

    # `axis=0` sums across polynomial terms, not data points
    # This avoids collapsing everything into a single scalar
    return np.sum([parameter_vector[i] * (x ** i) for i in range(degree + 1)], axis=0)
