import pandas as pd
from prophet import Prophet


type ProphetModel = Prophet


def apply_prophet_model(df: pd.DataFrame, x_column: str, y_column: str, model: ProphetModel) -> pd.DataFrame:
    df[y_column] = model.predict(df[x_column].rename("ds").to_frame())["yhat"].values
    return df

def train_prophet_model(df: pd.DataFrame, x_column: str, y_column: str) -> ProphetModel:
    model = Prophet()
    model.fit(df[[x_column, y_column]].rename(columns={
        x_column: "ds",
        y_column: "y",
    }))
    return model
