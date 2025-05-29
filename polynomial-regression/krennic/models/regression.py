import numpy as np


class RegressionModel:
    def build(self, x: np.ndarray, y: np.ndarray):
        pass

    def fit(self, x: float) -> float:
        pass


class LinearRegressionModel(RegressionModel):
    def build(self, x: np.ndarray, y: np.ndarray):
        assert len(x) == len(y)
        m = len(x)

        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_xx = np.sum(np.pow(x, 2))

        self.__a1 = (m * sum_xy - sum_x * sum_y) / (m * sum_xx - sum_x ** 2)
        self.__a0 = (sum_xx * sum_y - sum_xy * sum_x) / (m * sum_xx - sum_x ** 2)

    def fit(self, x: float) -> float:
        a1 = self.__a1
        a0 = self.__a0
        return a1 * x + a0


# TODO Impl Gaussian noise
class PolynomialRegressionModel(RegressionModel):
    def __init__(self, degree: int):
        super().__init__()

        self.__degree = degree

    def build(self, x: np.ndarray, y: np.ndarray):
        assert len(x) == len(y)

        # Naming conventions from https://en.wikipedia.org/wiki/Polynomial_regression
        cached_coefficients_of_design_matrix = np.array([np.sum(x ** p) for p in range(self.__degree * 2 + 1)])
        design_matrix = np.array([[cached_coefficients_of_design_matrix[i + j] for j in range(self.__degree + 1)] for i in range(self.__degree + 1)])
        response_vector = np.array([np.sum(y * (x ** i)) for i in range(self.__degree + 1)])
        parameter_vector = np.linalg.solve(design_matrix, response_vector)

        self.__polynomial_coefficients = parameter_vector

    def fit(self, x: float | np.ndarray) -> float | np.ndarray:
        # `axis=0` sums across polynomial terms, not data points
        # This avoid collapsing everything into a single scalar
        return np.sum([self.__polynomial_coefficients[i] * (x ** i) for i in range(self.__degree + 1)], axis=0)

    def  __repr__(self):
        return f"P(x) = {" + ".join([f"{self.__polynomial_coefficients[i]}*x^{i}" for i in range(self.__degree, -1, -1)])}"
