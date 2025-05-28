from matplotlib import pyplot as plt
import pandas as pd

from krennic.evaluation import evaluate_mae, evaluate_mse, evaluate_rmse
from krennic.normalization import apply_min_max_normalization
from krennic.polynomial_regression import apply_polynomial_regression_model, train_polynomial_regression_model


pd.options.mode.copy_on_write = True


def split_into_training_and_testing(df: pd.DataFrame, training_proportion: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    training_nrows = len(df) * training_proportion

    training_df = df.loc[:training_nrows - 1, :]
    testing_df = df.loc[training_nrows:, :]

    return training_df, testing_df


if __name__ == "__main__":
    DEGREE = 4

    df = (
        pd.read_csv(
            "resources/datasets/GlobalTemperatures.csv",
            usecols=["dt", "LandAverageTemperature"], parse_dates=["dt"],
        )
        .rename(columns={
            "dt": "timestamp",
            "LandAverageTemperature": "temperature",
        })
        .dropna()
        .pipe(apply_min_max_normalization, column="timestamp", new_column="timestamp-norm")
    )

    training_df, testing_df = df.pipe(split_into_training_and_testing, training_proportion=0.5)
    
    model = training_df.pipe(train_polynomial_regression_model, x_column="timestamp-norm", y_column="temperature", degree=DEGREE)
    training_df = training_df.pipe(apply_polynomial_regression_model, x_column="timestamp-norm", y_column="temperature-polyfit", model=model)
    testing_df = testing_df.pipe(apply_polynomial_regression_model, x_column="timestamp-norm", y_column="temperature-polyfit", model=model)
    
    mae = evaluate_mae(y=testing_df["temperature"], ŷ=testing_df["temperature-polyfit"])
    mse = evaluate_mse(y=testing_df["temperature"], ŷ=testing_df["temperature-polyfit"])
    rmse = evaluate_rmse(y=testing_df["temperature"], ŷ=testing_df["temperature-polyfit"])

    axes = df.plot(x="timestamp", y="temperature", color="blue", label="Temperature")
    training_df.plot(x="timestamp", y="temperature-polyfit", color="orange", label="Prediction", ax=axes)
    testing_df.plot(x="timestamp", y="temperature-polyfit", color="red", label="Prediction", ax=axes)
    axes.axvline(x=training_df["timestamp"].iloc[-1], color="gray", linestyle="--")

    plt.text(0.97, 0.03, f"MAE = {mae:.2f}\nMSE = {mse:.2f}\nRMSE = {rmse:.2f}",
             transform=axes.transAxes, horizontalalignment="right", verticalalignment="bottom",
             fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
