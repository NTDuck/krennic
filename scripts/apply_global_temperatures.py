from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

from krennic import apply
from krennic.models.regression import LinearRegressionModel, PolynomialRegressionModel


def load_df() -> pd.DataFrame:
    dataset = pd.read_csv(
        filepath_or_buffer="resources/datasets/GlobalTemperatures.csv",
        usecols=["dt", "LandAverageTemperature"], parse_dates=["dt"],
    )

    dataset.rename(columns={
        "dt": "timestamp",
        "LandAverageTemperature": "temperature",
    }, inplace=True)
    dataset["timestamp-int64"] = dataset["timestamp"].map(datetime.toordinal)

    return dataset


if __name__ == "__main__":
    pd.options.mode.copy_on_write = True
    
    df = load_df()
    # result = apply(df, x_column="timestamp-int64", y_column="temperature", regression_model=LinearRegressionModel())
    result = apply(df, x_column="timestamp-int64", y_column="temperature", regression_model=PolynomialRegressionModel(degree=100))

    print(result.testing_df.describe())
    print(f"MSE: {result.mse}, MAE: {result.mae}, RMSE: {result.rmse}")

    result.testing_df.plot(x="timestamp", y=["temperature", "predicted-temperature"])
    plt.show()
