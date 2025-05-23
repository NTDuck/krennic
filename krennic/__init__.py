import pandas as pd
import matplotlib.pyplot as plt


class Dataset(pd.DataFrame):
    TIMESTAMP_COLUMN_NAME = "timestamp"
    TEMPERATURE_COLUMN_NAME = "temperature"

    @property
    def timestamp(self):
        return self[self.TIMESTAMP_COLUMN_NAME]
    
    @property
    def temperature(self):
        return self[self.TEMPERATURE_COLUMN_NAME]
    

def load_global_temperatures_dataset() -> Dataset:
    dataset = pd.read_csv(
        filepath_or_buffer="resources/datasets/GlobalTemperatures.csv",
        usecols=["dt", "LandAverageTemperature"], parse_dates=["dt"],
    )

    dataset.rename(columns={
        "dt": Dataset.TIMESTAMP_COLUMN_NAME,
        "LandAverageTemperature": Dataset.TEMPERATURE_COLUMN_NAME,
    }, inplace=True)

    dataset.set_index(Dataset.TIMESTAMP_COLUMN_NAME, inplace=True)

    return dataset


if __name__ == "__main__":
    dataset = load_global_temperatures_dataset()
    dataset.plot()

    plt.show()
