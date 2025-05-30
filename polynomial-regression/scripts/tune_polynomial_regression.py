from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from krennic.evaluation import evaluate_mae, evaluate_mse, evaluate_rmse
from krennic.normalization import apply_min_max_normalization
from krennic.regression import apply_regression_model, train_polynomial_regression_model
from krennic.utils import split_into_training_and_testing


pd.options.mode.copy_on_write = True


def __evaluate(
    training_df: pd.DataFrame, testing_df: pd.DataFrame, degree: int
) -> dict:
    try:
        model = training_df.pipe(
            train_polynomial_regression_model,
            x_column="timestamp-norm",
            y_column="temperature",
            degree=degree,
        )
        training_df = training_df.pipe(
            apply_regression_model,
            x_column="timestamp-norm",
            y_column="temperature-polyfit",
            model=model,
        )
        testing_df = testing_df.pipe(
            apply_regression_model,
            x_column="timestamp-norm",
            y_column="temperature-polyfit",
            model=model,
        )

        mae = evaluate_mae(
            y=testing_df["temperature"], ŷ=testing_df["temperature-polyfit"]
        )
        mse = evaluate_mse(
            y=testing_df["temperature"], ŷ=testing_df["temperature-polyfit"]
        )
        rmse = evaluate_rmse(
            y=testing_df["temperature"], ŷ=testing_df["temperature-polyfit"]
        )

    except:
        mae, mse, rmse = np.nan, np.nan, np.nan

    return {
        "degree": degree,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
    }


if __name__ == "__main__":
    TRAINING_PROPORTION = 0.75

    # Hyperparameters
    DEGREES = range(1, 1000)

    df = (
        # pd.read_csv(
        #     "resources/datasets/GlobalTemperatures.csv",
        #     usecols=["dt", "LandAverageTemperature"], parse_dates=["dt"],
        # )
        # .rename(columns={
        #     "dt": "timestamp",
        #     "LandAverageTemperature": "temperature",
        # })
        pd.read_csv(
            "resources/datasets/hanoi-aqi-weather-data.csv",
            usecols=["Local Time", "Temperature"],
            parse_dates=["Local Time"],
        )
        .rename(
            columns={
                "Local Time": "timestamp",
                "Temperature": "temperature",
            }
        )
        .dropna()
        .pipe(
            apply_min_max_normalization, column="timestamp", new_column="timestamp-norm"
        )
    )

    training_df, testing_df = df.pipe(
        split_into_training_and_testing, training_proportion=TRAINING_PROPORTION
    )

    csv_path = f"resources/metrics/hanoi-aqi-weather-data.polynomial-regression.{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.csv"

    try:
        with ProcessPoolExecutor(max_workers=min(4, os.cpu_count())) as executor:
            futures = [
                executor.submit(
                    __evaluate,
                    training_df=training_df.copy(),
                    testing_df=testing_df.copy(),
                    degree=degree,
                )
                for degree in DEGREES
            ]

            with tqdm(total=len(futures), desc="Tuning") as pbar:
                for future in as_completed(futures):
                    pd.DataFrame([future.result()]).to_csv(
                        csv_path,
                        mode="a",
                        header=not os.path.exists(csv_path),
                        index=False,
                    )
                    pbar.update(1)

    except KeyboardInterrupt:
        executor.shutdown(wait=False, cancel_futures=True)

    # metrics_df = pd.DataFrame(metrics)
    # metrics_df.to_csv(f"resources/metrics/hanoi-aqi-weather-data.polynomial-regression.{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.csv")

    # row_with_min_mae = metrics_df.loc[metrics_df["mae"].idxmin()]
    # row_with_min_mse = metrics_df.loc[metrics_df["mse"].idxmin()]
    # row_with_min_rmse = metrics_df.loc[metrics_df["rmse"].idxmin()]

    # axes = metrics_df.plot(x="degree", y=["mae", "mse", "rmse"],
    #                        color=["orange", "blue", "red"], label=["MAE", "MSE", "RMSE"])
    # axes.text(0.97, 0.03, f"""Lowest MAE = {row_with_min_mae["mae"]:.2f} (n={row_with_min_mae["degree"]:.0f})
    #                         \nLowest MSE = {row_with_min_mse["mse"]:.2f} (n={row_with_min_mse["degree"]:.0f})
    #                         \nLowest RMSE = {row_with_min_rmse["rmse"]:.2f} (n={row_with_min_rmse["degree"]:.0f})""",
    #           transform=axes.transAxes, horizontalalignment="right", verticalalignment="bottom",
    #           fontsize=10, bbox=dict(facecolor="white", alpha=0.7))

    # plt.tight_layout()
    # plt.show()
