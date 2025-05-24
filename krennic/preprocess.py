import pandas as pd
from datetime import datetime
import numpy as np

def preprocess_data(path):
    data = pd.read_csv(path)
    n = int(0.15 * len(data))
    data = data[['UTC Time', 'Temperature']]
    data['UTC Time'] = pd.to_datetime(data['UTC Time'])
    data['Timestamp'] = data['UTC Time'].apply(lambda x: int(x.timestamp()) if pd.notnull(x) else None)
    data['Temperature_missing'] = data['Temperature']
    indices_to_remove = np.random.choice(data.index, size=n, replace=False)
    data.loc[indices_to_remove, 'Temperature_missing'] = np.nan
    data.to_csv('../data/weather_data.csv', index=False)

preprocess_data('../data/hanoi-aqi-weather-data.csv')