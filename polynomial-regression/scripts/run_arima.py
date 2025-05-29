import pandas as pd
from matplotlib import pyplot as plt

from krennic.arima import apply_arima_model, train_arima_model
from krennic.evaluation import evaluate_mae, evaluate_mse, evaluate_rmse
from krennic.utils import split_into_training_and_testing


pd.options.mode.copy_on_write = True


if __name__ == "__main__":
    TRAINING_PROPORTION = 0.75

    # Hyperparameters
    ARIMA_ORDER = (2, 1, 2)

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

    model = training_df.pipe(train_arima_model, y_column="temperature", order=ARIMA_ORDER)
    testing_df = testing_df.pipe(apply_arima_model, y_column="temperature-arima", model=model)

    mae = evaluate_mae(y=testing_df["temperature"], ŷ=testing_df["temperature-arima"])
    mse = evaluate_mse(y=testing_df["temperature"], ŷ=testing_df["temperature-arima"])
    rmse = evaluate_rmse(y=testing_df["temperature"], ŷ=testing_df["temperature-arima"])

    axes = df["temperature"].plot( color="blue", label="Temperature")
    testing_df["temperature-arima"].plot(color="red", label="Forecast", ax=axes)

    axes.axvline(x=training_df.index[-1], color="gray", linestyle="--")
    axes.text(0.97, 0.03, f"MAE = {mae:.2f}\nMSE = {mse:.2f}\nRMSE = {rmse:.2f}",
              transform=axes.transAxes, ha="right", va="bottom",
              fontsize=10, bbox=dict(facecolor="white", alpha=0.7))

    plt.tight_layout()
    plt.show()
