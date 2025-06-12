from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import os
import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from krennic.evaluation import evaluate_mae, evaluate_mse, evaluate_rmse
from krennic.normalization import apply_min_max_normalization
from krennic.regression import apply_regression_model, train_ridge_regression_model
from krennic.utils import split_into_training_and_testing


pd.options.mode.copy_on_write = True


def __evaluate(
    training_df: pd.DataFrame, testing_df: pd.DataFrame, degree: int, λ: float
) -> dict:
    start_time = time.perf_counter_ns()
    
    try:
        model = training_df.pipe(
            train_ridge_regression_model,
            x_column="timestamp-norm",
            y_column="temperature",
            degree=degree,
            λ=λ,
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

    elapsed_time = time.perf_counter_ns() - start_time

    return {
        "degree": degree,
        "λ": λ,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "time": elapsed_time,
    }


if __name__ == "__main__":
    TRAINING_PROPORTION = 0.75

    # Hyperparameters
    DEGREES = range(1, 100)
    λs = [10**i for i in range(-10, 10)]

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

    csv_path = f"resources/metrics/hanoi-aqi-weather-data.ridge-regression.{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.csv"

    try:
        with ProcessPoolExecutor(max_workers=min(4, os.cpu_count())) as executor:
            futures = [
                executor.submit(
                    __evaluate,
                    training_df=training_df.copy(),
                    testing_df=testing_df.copy(),
                    degree=degree,
                    λ=λ,
                )
                for degree in DEGREES
                for λ in λs
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
