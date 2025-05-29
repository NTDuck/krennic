import pandas as pd


def split_into_training_and_testing(df: pd.DataFrame, training_proportion: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    training_nrows = int(len(df) * training_proportion)

    training_df = df.iloc[:training_nrows, :]
    testing_df = df.iloc[training_nrows:, :]

    return training_df, testing_df
