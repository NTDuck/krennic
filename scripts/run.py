from typing import Optional
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def toordinal(df: pd.DataFrame, column: str) -> pd.DataFrame:
    pass

def normalize_z_score(df: pd.DataFrame, column: str, new_column: Optional[str]) -> pd.DataFrame:
    new_column = new_column or column
    df[new_column] = __normalize_z_score(df[column])
    return df

def __normalize_z_score(x: np.ndarray) -> np.ndarray:
    return (x - np.mean(x)) / np.std(x)

def normalize_min_max(df: pd.DataFrame, column: str, new_column: Optional[str]) -> pd.DataFrame:
    new_column = new_column or column
    df[new_column] = __normalize_min_max(df[column])
    return df

def __normalize_min_max(x: np.ndarray) -> np.ndarray:
    x_max = np.max(x)
    x_min = np.min(x)
    return (x - x_min) / (x_max - x_min)

def normalize_abs_max(df: pd.DataFrame, column: str, new_column: Optional[str]) -> pd.DataFrame:
    new_column = new_column or column
    df[new_column] = __normalize_abs_max(df[column])
    pass

def __normalize_abs_max(x: np.ndarray) -> np.ndarray:
    x_abs_max = np.max(np.abs(x))
    return (x - x_abs_max) / x_abs_max

def normalize_robust(df: pd.DataFrame, column: str, new_column: Optional[str]) -> pd.DataFrame:
    new_column = new_column or column
    df[new_column] = __normalize_robust(df[column])
    pass

def __normalize_robust(x: np.ndarray) -> np.ndarray:
    x_median = np.median(x)
    x_q75, x_q25 = np.percentile(x, [75, 25])
    x_iqr = x_q75 - x_q25
    return (x - x_median) / x_iqr


pd.options.mode.copy_on_write = True


if __name__ == "__main__":
    df = (
        pd.read_csv(
            "resources/datasets/GlobalTemperatures.csv",
            usecols=["dt", "LandAverageTemperature"], parse_dates=["dt"],
        )
        .rename(columns={
            "dt": "timestamp",
            "LandAverageTemperature": "temperature",
        })
        .dropna()
        .pipe(normalize_z_score, column="timestamp")
    )

    df.plot(x="timestamp", y="temperature")
    plt.show()
