import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_global_temperatures_dataset() -> pd.DataFrame:
    dataset = pd.read_csv(
        filepath_or_buffer="resources/datasets/GlobalTemperatures.csv",
        usecols=["dt", "LandAverageTemperature"], parse_dates=["dt"],
    )

    dataset.rename(columns={
        "dt": "timestamp",
        "LandAverageTemperature": "temperature",
    }, inplace=True)

    dataset.set_index("timestamp", inplace=True)

    return dataset


if __name__ == "__main__":
    pd.options.mode.copy_on_write = True

    dataset = load_global_temperatures_dataset()

    TRAINING_PROPORTION = 0.75
    cutoff = int(len(dataset) * TRAINING_PROPORTION)

    training_dataset = dataset.iloc[:cutoff].dropna(subset=["temperature"])
    testing_dataset = dataset.iloc[cutoff:].dropna(subset=["temperature"])

    regression_model = 
    a, b = least_squares_fit_linear(
        x=training_dataset.index.map(pd.Timestamp.toordinal).to_numpy(),
        y=training_dataset["temperature"].to_numpy(),
    )

    print(f"Testing dataset: {a}x + {b}")

    testing_dataset["predicted-temperature"] = a * testing_dataset.index.map(pd.Timestamp.toordinal) + b
    testing_dataset["local-error"] = np.abs(testing_dataset["predicted-temperature"].to_numpy() - testing_dataset["temperature"].to_numpy())

    print(testing_dataset)
    # plt.show()
