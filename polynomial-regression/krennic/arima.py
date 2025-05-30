from typing import Optional

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults


type ArimaModel = ARIMAResults


def apply_arima_model(
    df: pd.DataFrame, y_column: str, model: ArimaModel, steps: Optional[int] = None
) -> pd.DataFrame:
    steps = steps or len(df)
    df[y_column] = model.forecast(steps=steps)
    return df


def train_arima_model(
    df: pd.DataFrame, y_column: str, order: tuple[int, int, int]
) -> ArimaModel:
    return ARIMA(df[y_column], order=order).fit()
