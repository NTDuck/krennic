from matplotlib import pyplot as plt
import pandas as pd

from krennic.normalization import normalize_min_max


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
        .pipe(normalize_min_max, column="timestamp")
    )

    df.plot(x="timestamp", y="temperature")
    plt.show()
