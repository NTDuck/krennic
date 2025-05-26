import argparse
from datetime import datetime
import numpy as np
import pandas as pd
from krennic.evaluators import MaeEvaluator, MseEvaluator, ResidualEvaluator, RmseEvaluator
from krennic.models.regression import PolynomialRegressionModel, RegressionModel
from scripts.utils import load_global_temperatures_df, load_hanoi_aqi_weather_data_df


def apply(df: pd.DataFrame, x_column: str, y_column: str, degrees: list[int], training_proportion: float) -> pd.DataFrame:
    pd.options.mode.copy_on_write = True

    assert x_column in df
    assert y_column in df

    df.dropna(subset=[x_column, y_column], inplace=True)

    training_rows_count = int(len(df) * training_proportion)
    training_df = df.iloc[:training_rows_count]
    testing_df = df.iloc[training_rows_count:]

    training_df_x = training_df[x_column].to_numpy()
    training_df_y = training_df[y_column].to_numpy()
    testing_df_actual_y = testing_df[y_column].to_numpy()

    rows = []

    for degree in degrees:
        regression_model = PolynomialRegressionModel(degree=degree)
        regression_model.build(x=training_df_x, y=training_df_y)
        testing_df_predicted_y = regression_model.fit(testing_df[x_column])

        residuals = ResidualEvaluator().evaluate(predicted=testing_df_predicted_y, actual=testing_df_actual_y)

        mse = MseEvaluator().evaluate(predicted=testing_df_predicted_y, actual=testing_df_actual_y)
        mae = MaeEvaluator().evaluate(predicted=testing_df_predicted_y, actual=testing_df_actual_y)
        rmse = RmseEvaluator().evaluate(predicted=testing_df_predicted_y, actual=testing_df_actual_y)

        rows.append({
            "polynomial-regression-model-degree": degree,
            "residuals-min": np.min(residuals),
            "residuals-max": np.max(residuals),
            "residuals-µ": np.mean(residuals),
            "residuals-σ": np.std(residuals),
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--degrees', nargs='+', type=int)
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

    df["timestamp-int64"] = df["timestamp"].map(datetime.toordinal)
    df = apply(df, x_column="timestamp-int64", y_column="temperature", degrees=args.degrees, training_proportion=args.training_proportion)
    
    df.to_csv(f"out/compare-{args.dataset}")
