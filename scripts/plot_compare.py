from matplotlib import pyplot as plt
import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv("out/compare-hanoi-aqi-weather-data.csv")

    df.plot(x="polynomial-regression-model-degree", y=["residuals-min", "residuals-max", "residuals-µ", "residuals-σ"])
    df.plot(x="polynomial-regression-model-degree", y=["mse", "mae", "rmse"])

    metrics = [
        "residuals-min",
        "residuals-max",
        "residuals-µ",
        "residuals-σ",
        "mse",
        "mae",
        "rmse",
    ]

    for metric in metrics:
        best_row = df.loc[df[metric].idxmin()]
        degree = best_row["polynomial-regression-model-degree"]
        value = best_row[metric]
        print(f"Best degree for {metric}: {degree} (value = {value})")

    # for metric in metrics:
    #     df.plot(x="polynomial-regression-model-degree", y=metric)
    plt.show()
