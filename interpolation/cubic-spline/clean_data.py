import pandas as pd
import numpy as np

data = pd.read_csv("hanoi-aqi-weather-data.csv")
data = data[["Local Time", "Temperature"]]

num_missing = int(0.05 * len(data))
missing_indices = np.random.choice(data.index, size=num_missing, replace=False)

missing_data = data.loc[missing_indices, ["Local Time", "Temperature"]]
missing_data.to_csv("missing.csv", index=False)

data.loc[missing_indices, "Temperature"] = np.nan

data.to_csv("real_data.csv", index=False)

