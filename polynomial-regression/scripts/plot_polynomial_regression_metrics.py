from matplotlib import pyplot as plt
import pandas as pd


if __name__ == "__main__":
    metrics_df = pd.read_csv("resources/metrics/hanoi-aqi-weather-data.polynomial-regression.2025-06-12 17-33-52.csv")

    row_with_min_mae = metrics_df.loc[metrics_df["mae"].idxmin()]
    row_with_min_mse = metrics_df.loc[metrics_df["mse"].idxmin()]
    row_with_min_rmse = metrics_df.loc[metrics_df["rmse"].idxmin()]

    axes = metrics_df.plot(
        x="degree",
        y=["mae", "mse", "rmse"],
        # y="time",
        color=["orange", "blue", "red"],
        # color="blue",
        label=["MAE", "MSE", "RMSE"],
        # label="time (ns)",
    )
    axes.text(
        0.97,
        0.03,
        f"""Lowest MAE = {row_with_min_mae["mae"]:.2f} (n={row_with_min_mae["degree"]:.0f})
                            \nLowest MSE = {row_with_min_mse["mse"]:.2f} (n={row_with_min_mse["degree"]:.0f})
                            \nLowest RMSE = {row_with_min_rmse["rmse"]:.2f} (n={row_with_min_rmse["degree"]:.0f})""",
        # f"""Lowest time = {row_with_min_mae["time"]:.0e} ns (n={row_with_min_mae["degree"]:.0f})""",
        transform=axes.transAxes,
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7),
    )

    plt.tight_layout()
    plt.show()
