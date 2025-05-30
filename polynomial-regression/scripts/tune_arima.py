from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from krennic.arima import apply_arima_model, train_arima_model
from krennic.evaluation import evaluate_mae, evaluate_mse, evaluate_rmse
from krennic.utils import split_into_training_and_testing


pd.options.mode.copy_on_write = True

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def __evaluate(
    training_df: pd.DataFrame, testing_df: pd.DataFrame, order: tuple[int, int, int]
) -> dict:
    try:
        model = training_df.pipe(train_arima_model, y_column="temperature", order=order)
        testing_df = testing_df.pipe(
            apply_arima_model, y_column="temperature-arima", model=model
        )

        mae = evaluate_mae(
            y=testing_df["temperature"], ŷ=testing_df["temperature-arima"]
        )
        mse = evaluate_mse(
            y=testing_df["temperature"], ŷ=testing_df["temperature-arima"]
        )
        rmse = evaluate_rmse(
            y=testing_df["temperature"], ŷ=testing_df["temperature-arima"]
        )

    except:
        mae, mse, rmse = np.nan, np.nan, np.nan

    return {
        "p": order[0],
        "d": order[1],
        "q": order[2],
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
    }


if __name__ == "__main__":
    TRAINING_PROPORTION = 0.75

    # Hyperparameters
    __MAX_ORDER = 10
    ORDERS = [
        (p, d, q)
        for p in range(1, __MAX_ORDER)
        for d in range(1, __MAX_ORDER)
        for q in range(1, __MAX_ORDER)
    ]

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
        .set_index("timestamp")
        # .asfreq("MS")
        .asfreq("h")
    )

    training_df, testing_df = df.pipe(
        split_into_training_and_testing, training_proportion=TRAINING_PROPORTION
    )

    csv_path = f"resources/metrics/hanoi-aqi-weather-data.arima.{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.csv"

    try:
        with ProcessPoolExecutor(max_workers=min(4, os.cpu_count())) as executor:
            futures = [
                executor.submit(
                    __evaluate,
                    training_df=training_df.copy(),
                    testing_df=testing_df.copy(),
                    order=order,
                )
                for order in ORDERS
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
