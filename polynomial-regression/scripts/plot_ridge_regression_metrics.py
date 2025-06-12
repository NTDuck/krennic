from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


if __name__ == "__main__":
    metrics_df = pd.read_csv("resources/metrics/hanoi-aqi-weather-data.ridge-regression.2025-06-12 17-54-11.csv")

    row_with_min_rmse = metrics_df.loc[metrics_df["mae"].idxmin()]

    axes = sns.heatmap(
        data=metrics_df.pivot(index="degree", columns="λ", values="time"),
        annot=False,
        fmt=".0e",
        cmap="coolwarm_r",
    )
    plt.title("MAE Heatmap")
    plt.xlabel("λ")
    plt.ylabel("Degree")

    axes.text(
        0.97,
        0.03,
        f"""Lowest MAE = {row_with_min_rmse["mae"]:.2f} (n={row_with_min_rmse["degree"]:.0f}, λ={row_with_min_rmse["λ"]})""",
        transform=axes.transAxes,
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7),
    )

    plt.tight_layout()
    plt.show()
