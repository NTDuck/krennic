import pandas as pd

from .evaluators import MaeEvaluator, MseEvaluator, ResidualEvaluator, RmseEvaluator
from .models.regression import RegressionModel


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

    training_df_x = training_df[x_column].to_numpy()
    training_df_y = training_df[y_column].to_numpy()
    regression_model = regression_model_cls(x=training_df_x, y=training_df_y)

    testing_df[f"predicted-{y_column}"] = regression_model.fit(testing_df[x_column])
    testing_df_predicted_y = testing_df[f"predicted-{y_column}"]
    testing_df_actual_y = testing_df[y_column]

    testing_df["residual"] = ResidualEvaluator().evaluate(predicted=testing_df_predicted_y, actual=testing_df_actual_y)
    
    mse = MseEvaluator().evaluate(predicted=testing_df_predicted_y, actual=testing_df_actual_y)
    mae = MaeEvaluator().evaluate(predicted=testing_df_predicted_y, actual=testing_df_actual_y)
    rmse = RmseEvaluator().evaluate(predicted=testing_df_predicted_y, actual=testing_df_actual_y)

    return Result(training_df, testing_df, mse, mae, rmse)
