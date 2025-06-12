from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


if __name__ == "__main__":
    metrics_df = pd.read_csv("resources/metrics/hanoi-aqi-weather-data.ridge-regression.2025-06-12 16-45-28.csv")

    row_with_min_rmse = metrics_df.loc[metrics_df["time"].idxmin()]

    axes = sns.heatmap(
        data=metrics_df.pivot(index="degree", columns="λ", values="time"),
        annot=True,
        fmt=".2f",
        cmap="coolwarm_r",
    )
    plt.title("Time Heatmap")
    plt.xlabel("λ")
    plt.ylabel("Degree")

    axes.text(
        0.97,
        0.03,
        f"""Lowest time = {row_with_min_rmse["time"]:.2f} (n={row_with_min_rmse["degree"]:.0f}, λ={row_with_min_rmse["λ"]})""",
        transform=axes.transAxes,
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7),
    )

    plt.tight_layout()
    plt.show()
