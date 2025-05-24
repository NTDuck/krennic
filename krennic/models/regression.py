import numpy as np


class RegressionModel:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        assert len(x) == len(y)
        pass

    def fit(self, x: float) -> float:
        pass


class LinearRegressionModel(RegressionModel):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        super().__init__(x, y)
        n = len(x)

        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_xx = np.sum(np.pow(x, 2))    

        self.__a = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - np.pow(sum_x, 2))
        self.__b = (sum_xx * sum_y - sum_xy * sum_x) / (n * sum_xx - np.pow(sum_x, 2))

    def fit(self, x: float) -> float:
        return self.__a * x + self.__b


class QuadraticRegressionModel(RegressionModel):
    pass
