from typing import Optional
import numpy as np
import pandas as pd


def apply_z_score_normalization(df: pd.DataFrame, column: str, new_column: Optional[str] = None) -> pd.DataFrame:
    new_column = new_column or column
    df[new_column] = __apply_z_score_normalization(df[column])
    return df

def __apply_z_score_normalization(x: np.ndarray) -> np.ndarray:
    return (x - np.mean(x)) / np.std(x)

def apply_min_max_normalization(df: pd.DataFrame, column: str, new_column: Optional[str] = None) -> pd.DataFrame:
    new_column = new_column or column
    df[new_column] = __apply_min_max_normalization(df[column])
    return df

def __apply_min_max_normalization(x: np.ndarray) -> np.ndarray:
    x_max = np.max(x)
    x_min = np.min(x)
    return (x - x_min) / (x_max - x_min)

def apply_abs_max_normalization(df: pd.DataFrame, column: str, new_column: Optional[str] = None) -> pd.DataFrame:
    new_column = new_column or column
    df[new_column] = __apply_abs_max_normalization(df[column])
    pass

def __apply_abs_max_normalization(x: np.ndarray) -> np.ndarray:
    x_abs_max = np.max(np.abs(x))
    return (x - x_abs_max) / x_abs_max

def apply_robust_normalization(df: pd.DataFrame, column: str, new_column: Optional[str] = None) -> pd.DataFrame:
    new_column = new_column or column
    df[new_column] = __apply_robust_normalization(df[column])
    pass

def __apply_robust_normalization(x: np.ndarray) -> np.ndarray:
    x_median = np.median(x)
    x_q75, x_q25 = np.percentile(x, [75, 25])
    x_iqr = x_q75 - x_q25
    return (x - x_median) / x_iqr
