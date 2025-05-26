import pandas as pd


def load_global_temperatures_df() -> pd.DataFrame:
    dataset = pd.read_csv(
        "resources/datasets/GlobalTemperatures.csv",
        usecols=["dt", "LandAverageTemperature"], parse_dates=["dt"],
    )

    dataset.rename(columns={
        "dt": "timestamp",
        "LandAverageTemperature": "temperature",
    }, inplace=True)

    return dataset

def load_hanoi_aqi_weather_data_df() -> pd.DataFrame:
    dataset = pd.read_csv(
        "resources/datasets/hanoi-aqi-weather-data.csv",
        usecols=["Local Time", "Temperature"], parse_dates=["Local Time"],
    )

    dataset.rename(columns={
        "Local Time": "timestamp",
        "Temperature": "temperature",
    }, inplace=True)

    return dataset
