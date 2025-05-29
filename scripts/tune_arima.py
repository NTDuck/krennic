import warnings
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from krennic.arima import apply_arima_model, train_arima_model
from krennic.evaluation import evaluate_mae, evaluate_mse, evaluate_rmse
from krennic.utils import split_into_training_and_testing


pd.options.mode.copy_on_write = True

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":
    TRAINING_PROPORTION = 0.75

    # Hyperparameters
    PS = range(1, 3)
    DS = range(1, 3)
    QS = range(1, 3)
    
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
            usecols=["Local Time", "Temperature"], parse_dates=["Local Time"],
        )
        .rename(columns={
            "Local Time": "timestamp",
            "Temperature": "temperature",
        })
        .dropna()
        .set_index("timestamp")
        # .asfreq("MS")
        .asfreq("h")
    )

    training_df, testing_df = df.pipe(split_into_training_and_testing, training_proportion=TRAINING_PROPORTION)

    metrics = []

    for p in PS:
        for d in DS:
            for q in QS:
                training_df = training_df.copy()
                testing_df = testing_df.copy()

                model = training_df.pipe(train_arima_model, y_column="temperature", order=(p, d, q))
                testing_df = testing_df.pipe(apply_arima_model, y_column="temperature-arima", model=model)

                mae = evaluate_mae(y=testing_df["temperature"], ŷ=testing_df["temperature-arima"])
                mse = evaluate_mse(y=testing_df["temperature"], ŷ=testing_df["temperature-arima"])
                rmse = evaluate_rmse(y=testing_df["temperature"], ŷ=testing_df["temperature-arima"])

                metrics.append({
                    "p": p,
                    "d": d,
                    "q": q,
                    "mae": evaluate_mae(y=testing_df["temperature"], ŷ=testing_df["temperature-arima"]),
                    "mse": evaluate_mse(y=testing_df["temperature"], ŷ=testing_df["temperature-arima"]),
                    "rmse": evaluate_rmse(y=testing_df["temperature"], ŷ=testing_df["temperature-arima"]),
                })

    metrics_df = pd.DataFrame(metrics)

    row_with_min_mae = metrics_df.loc[metrics_df["mae"].idxmin()]
    row_with_min_mse = metrics_df.loc[metrics_df["mse"].idxmin()]
    row_with_min_rmse = metrics_df.loc[metrics_df["rmse"].idxmin()]

    print(f"""Lowest MAE = {row_with_min_mae["mae"]:.2f} (p={row_with_min_mae["p"]:.0f}, d={row_with_min_mae["d"]:.0f}, q={row_with_min_mae["q"]:.0f})\
              Lowest MSE = {row_with_min_mse["mse"]:.2f} (p={row_with_min_mse["p"]:.0f}, d={row_with_min_mse["d"]:.0f}, q={row_with_min_mse["q"]:.0f})\
              Lowest RMSE = {row_with_min_rmse["rmse"]:.2f} (p={row_with_min_rmse["p"]:.0f}, d={row_with_min_rmse["d"]:.0f}, q={row_with_min_rmse["q"]:.0f})""")
