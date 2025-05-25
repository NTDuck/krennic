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

        self.__a_1 = (m * sum_xy - sum_x * sum_y) / (m * sum_xx - np.pow(sum_x, 2))
        self.__a_0 = (sum_xx * sum_y - sum_xy * sum_x) / (m * sum_xx - np.pow(sum_x, 2))

    def fit(self, x: float) -> float:
        return self.__a_1 * x + self.__a_0


class PolynomialRegressionModel(RegressionModel):
    def __init__(self, degree: int):
        super().__init__()

        self.__degree = degree

    def build(self, x: np.ndarray, y: np.ndarray):
        assert len(x) == len(y)
        n = self.__degree + 1

        cached_coeffs = np.array([np.sum(x ** p) for p in range(n << 1)])
        a = np.array([[cached_coeffs[i + j] for j in range(n)] for i in range(n)])
        b = np.array([np.sum(y * (x ** i)) for i in range(n)])
        self.__coeffs = np.linalg.solve(a, b)

    def fit(self, x: float | np.ndarray) -> float | np.ndarray:
        n = self.__degree + 1
        # `axis=0` sums across polynomial terms, not data points
        # This avoid collapsing everything into a single scalar
        return np.sum([self.__coeffs[i] * (x ** i) for i in range(n)], axis=0)

    def  __repr__(self):
        return super().__repr__()
