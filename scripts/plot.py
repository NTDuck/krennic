import argparse
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import pandas as pd

from krennic.models.regression import PolynomialRegressionModel, RegressionModel
from scripts.utils import load_global_temperatures_df, load_hanoi_aqi_weather_data_df


def apply(df: pd.DataFrame, x_column: str, y_column: str, regression_model: RegressionModel, training_proportion: float) -> pd.DataFrame:
    pd.options.mode.copy_on_write = True

    assert x_column in df
    assert y_column in df

    df.dropna(subset=[x_column, y_column], inplace=True)
    # df = df.tail(72)

    training_rows_count = int(len(df) * training_proportion)
    training_df = df.iloc[:training_rows_count]
    testing_df = df.iloc[training_rows_count:]

    training_df_x = training_df[x_column].to_numpy()
    training_df_y = training_df[y_column].to_numpy()
    regression_model.build(x=training_df_x, y=training_df_y)

    testing_df[f"predicted-{y_column}"] = regression_model.fit(testing_df[x_column])

    df = pd.concat([training_df, testing_df])
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=[
        "GlobalTemperatures.csv",
        "hanoi-aqi-weather-data.csv",
    ])
    parser.add_argument("--training-proportion", type=float, required=False, default=0.9)

    args = parser.parse_args()

    if args.dataset == "GlobalTemperatures.csv":
        df = load_global_temperatures_df()
    elif args.dataset == "hanoi-aqi-weather-data.csv":
        df = load_hanoi_aqi_weather_data_df()

    regression_model = PolynomialRegressionModel(degree=args.degree)

    df["timestamp-int64"] = df["timestamp"].map(datetime.toordinal)
    df["timestamp-int64-norm"] = (df["timestamp-int64"] - np.mean(df["timestamp-int64"])) / np.std(df["timestamp-int64"])
    df["temperature-norm"] = df["temperature"] - np.mean(df["temperature"]) / np.std(df["temperature"])

    df = apply(df, x_column="timestamp-int64-norm", y_column="temperature-norm", regression_model=regression_model, training_proportion=args.training_proportion)
    del df["timestamp-int64"]
    
    df.plot(x="timestamp", y=["predicted-temperature-norm", "temperature-norm"])
    plt.tight_layout()
    plt.show()
