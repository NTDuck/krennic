from matplotlib import pyplot as plt
import pandas as pd

from krennic.evaluation import evaluate_mae, evaluate_mse, evaluate_rmse
from krennic.normalization import apply_min_max_normalization
from krennic.regression import apply_regression_model, train_ridge_regression_model
from krennic.utils import split_into_training_and_testing


pd.options.mode.copy_on_write = True


if __name__ == "__main__":
    TRAINING_PROPORTION = 0.75

    # Hyperparameters
    DEGREE = 2
    λ = 10

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

    model = training_df.pipe(
        train_ridge_regression_model,
        x_column="timestamp-norm",
        y_column="temperature",
        degree=DEGREE,
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

    mae = evaluate_mae(y=testing_df["temperature"], ŷ=testing_df["temperature-polyfit"])
    mse = evaluate_mse(y=testing_df["temperature"], ŷ=testing_df["temperature-polyfit"])
    rmse = evaluate_rmse(
        y=testing_df["temperature"], ŷ=testing_df["temperature-polyfit"]
    )

    axes = df.plot(x="timestamp", y="temperature", color="blue", label="Temperature")
    training_df.plot(
        x="timestamp",
        y="temperature-polyfit",
        color="orange",
        label="Forecast (training)",
        ax=axes,
    )
    testing_df.plot(
        x="timestamp",
        y="temperature-polyfit",
        color="red",
        label="Forecast (testing)",
        ax=axes,
    )

    axes.axvline(x=training_df["timestamp"].iloc[-1], color="gray", linestyle="--")
    axes.text(
        0.97,
        0.03,
        f"MAE = {mae:.2f}\nMSE = {mse:.2f}\nRMSE = {rmse:.2f}",
        transform=axes.transAxes,
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7),
    )

    plt.tight_layout()
    plt.show()
