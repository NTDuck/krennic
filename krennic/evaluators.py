from typing import Any
import numpy as np


class ErrorEvaluator:
    def evaluate(self, predicted: np.ndarray, actual: np.ndarray) -> Any:
        pass


class ResidualEvaluator(ErrorEvaluator):
    def evaluate(self, predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
        return np.abs(predicted - actual)
    
type LocalErrorEvaluator = ResidualEvaluator


class GlobalErrorEvaluator(ErrorEvaluator):
    def evaluate(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        pass
    

class MseEvaluator(GlobalErrorEvaluator):
    def evaluate(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        residual = ResidualEvaluator().evaluate(predicted, actual)
        return np.mean(np.pow(residual, 2))
    

class MaeEvaluator(GlobalErrorEvaluator):
    def evaluate(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        residual = ResidualEvaluator().evaluate(predicted, actual)
        return np.mean(np.pow(residual, 2))


class RmseEvaluator(GlobalErrorEvaluator):
    def evaluate(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        mse = MseEvaluator().evaluate(predicted, actual)
        return np.sqrt(mse)

