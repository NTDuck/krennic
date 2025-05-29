import numpy as np


def evaluate_residuals(y: np.ndarray, ŷ: np.ndarray) -> np.ndarray:
    return np.abs(y - ŷ)

def evaluate_mse(y: np.ndarray, ŷ: np.ndarray) -> float:
    residuals = evaluate_residuals(y, ŷ)
    return np.mean(np.pow(residuals, 2))

def evaluate_mae(y: np.ndarray, ŷ: np.ndarray) -> float:
    residuals = evaluate_residuals(y, ŷ)
    return np.mean(np.abs(residuals))

def evaluate_rmse(y: np.ndarray, ŷ: np.ndarray) -> float:
    mse = evaluate_mse(y, ŷ)
    return np.sqrt(mse)
