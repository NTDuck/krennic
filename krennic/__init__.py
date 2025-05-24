import pandas as pd
import matplotlib.pyplot as plt

from evaluators import MaeEvaluator, MseEvaluator, ResidualEvaluator, RmseEvaluator
from models.regression import LinearRegressionModel, RegressionModel


class Result:
    def __init__(self, training_df: pd.DataFrame, testing_df: pd.DataFrame, mse: float, mae: float, rmse: float):
        self.training_df = training_df
        self.testing_df = testing_df
        self.mse = mse
        self.mae = mae
        self.rmse = rmse


def apply(df: pd.DataFrame, x_column: str, y_column: str, regression_model_cls: type[RegressionModel]) -> Result:
    assert x_column in df
    assert y_column in df

    TRAINING_PROPORTION = 0.75
    cutoff = int(len(df) * TRAINING_PROPORTION)

    df.dropna(subset=[x_column, y_column], inplace=True)

    training_df = df.iloc[:cutoff]
    testing_df = df.iloc[cutoff:]

    x = training_df[x_column].to_numpy()
    y = training_df[y_column].to_numpy()
    regression_model = regression_model_cls(x, y)

    testing_df[f"predicted-{y_column}"] = regression_model.fit(testing_df[x_column])
    predicted_y = testing_df[f"predicted-{y_column}"]

    testing_df["residual"] = ResidualEvaluator().evaluate(predicted=predicted_y, actual=y)
    
    mse = MseEvaluator().evaluate(predicted=predicted_y, actual=y)
    mae = MaeEvaluator().evaluate(predicted=predicted_y, actual=y)
    rmse = RmseEvaluator().evaluate(predicted=predicted_y, actual=y)

    return Result(training_df, testing_df, mse, mae, rmse)


def load_global_temperatures_df() -> pd.DataFrame:
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
    
    df = load_global_temperatures_df()
    df.plot()
    # result = apply(df, x_column="timestamp", y_column="temperature", regression_model_cls=LinearRegressionModel)

    # result.testing_df.plot()
    plt.show()
