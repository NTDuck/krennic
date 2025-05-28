from matplotlib import pyplot as plt
import pandas as pd

from krennic.evaluation import evaluate_mae, evaluate_mse, evaluate_rmse
from krennic.normalization import apply_min_max_normalization
from krennic.polynomial_regression import fit_polynomial_regression


pd.options.mode.copy_on_write = True


if __name__ == "__main__":
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
        .pipe(fit_polynomial_regression, x_column="timestamp-norm", y_column="temperature", new_y_column="temperature-polyfit", degree=444)
    )

    ax = df.plot(x="timestamp", y=["temperature", "temperature-polyfit"], grid=True)

    mae = evaluate_mae(y=df["temperature"], ŷ=df["temperature-polyfit"])
    mse = evaluate_mse(y=df["temperature"], ŷ=df["temperature-polyfit"])
    rmse = evaluate_rmse(y=df["temperature"], ŷ=df["temperature-polyfit"])

    plt.text(0.97, 0.03, f"MAE = {mae:.2f}\nMSE = {mse:.2f}\nRMSE = {rmse:.2f}",
             transform=ax.transAxes, horizontalalignment="right", verticalalignment="bottom",
             fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
