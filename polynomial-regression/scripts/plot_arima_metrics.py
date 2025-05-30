import pandas as pd


if __name__ == "__main__":
    metrics_df = pd.read_csv("resources/metrics/hanoi-aqi-weather-data.arima.2025-05-30 13-20-51.csv")

    row_with_min_mae = metrics_df.loc[metrics_df["mae"].idxmin()]
    row_with_min_mse = metrics_df.loc[metrics_df["mse"].idxmin()]
    row_with_min_rmse = metrics_df.loc[metrics_df["rmse"].idxmin()]

    print(f"Lowest MAE = {row_with_min_mae["mae"]:.2f} (p={row_with_min_mae["p"]:.0f}, d={row_with_min_mae["d"]:.0f}, q={row_with_min_mae["q"]:.0f})")
    print(f"Lowest MSE = {row_with_min_mse["mse"]:.2f} (p={row_with_min_mse["p"]:.0f}, d={row_with_min_mse["d"]:.0f}, q={row_with_min_mse["q"]:.0f})")
    print(f"Lowest RMSE = {row_with_min_rmse["rmse"]:.2f} (p={row_with_min_rmse["p"]:.0f}, d={row_with_min_rmse["d"]:.0f}, q={row_with_min_rmse["q"]:.0f})")
