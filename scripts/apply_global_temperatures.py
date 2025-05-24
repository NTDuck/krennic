import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from krennic import apply
from krennic.models.regression import LinearRegressionModel


def load_df() -> pd.DataFrame:
    dataset = pd.read_csv(
        filepath_or_buffer="resources/datasets/GlobalTemperatures.csv",
        usecols=["dt", "LandAverageTemperature"], parse_dates=["dt"],
    )

    dataset.rename(columns={
        "dt": "timestamp",
        "LandAverageTemperature": "temperature",
    }, inplace=True)
    dataset["timestamp-int64"] = dataset["timestamp"].astype(np.int64)

    # dataset.set_index("timestamp", inplace=True)

    return dataset


if __name__ == "__main__":
    pd.options.mode.copy_on_write = True
    
    df = load_df()
    df.plot()
    result = apply(df, x_column="timestamp-int64", y_column="temperature", regression_model_cls=LinearRegressionModel)

    result.testing_df.plot()
    plt.show()
