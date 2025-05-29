import numpy as np
import pandas as pd

# Lagrange Interpolation Functions
def lagrange_interpolate(x, y, x0):
    total = 0.0
    n = len(x)
    for i in range(n):
        term = y[i]
        for j in range(n):
            if j != i:
                term *= (x0 - x[j]) / (x[i] - x[j])
        total += term
    return total

# Newton Interpolation Functions
def newton_divided_differences(x, y):
    x = np.array(x, dtype=float)
    coef = np.array(y, dtype=float)
    n = len(x)
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (x[j:n] - x[0:n-j])
    return coef

def newton_interpolate(x, coef, x0):
    n = len(coef)
    result = coef[-1]
    for i in range(n - 2, -1, -1):
        result = result * (x0 - x[i]) + coef[i]
    return result

def select_neighbors(i, temperatures, timestamps, max_missing=2):
    n = len(temperatures)

    if np.isnan(temperatures[i]):
        # Các tổ hợp (trái, phải) để lấy đủ 4 điểm (hoặc gần đó)
        for left, right in [(2, 2), (3, 1), (1, 3)]:
            start = i - left
            end = i + 1 + right
            if start >= 0 and end <= n:
                indices = list(range(i - left, i)) + list(range(i + 1, i + 1 + right))
                if all(not np.isnan(temperatures[idx]) for idx in indices):
                    return indices

    # Trường hợp mất liên tiếp 2 điểm: i và i+1
    if i + 1 < n and np.isnan(temperatures[i]) and np.isnan(temperatures[i + 1]):
        for left, right in [(3, 1), (2, 2), (1, 3)]:
            start = i - left
            end = i + 2 + right
            if start >= 0 and end <= n:
                indices = list(range(i - left, i)) + list(range(i + 2, i + 2 + right))
                if all(not np.isnan(temperatures[idx]) for idx in indices):
                    return indices

    return None


def lagrange_fill_missing(timestamps, temperatures):
    temperatures = np.array(temperatures, dtype=np.float64)
    timestamps = np.array(timestamps)
    result = temperatures.copy()
    n = len(temperatures)

    for i in range(n):
        if not np.isnan(temperatures[i]):
            continue

        indices = select_neighbors(i, result, timestamps)
        if indices is None:
            continue

        x_vals = [timestamps[j] for j in indices]
        y_vals = [result[j] for j in indices]

        try:
            result[i] = round(lagrange_interpolate(x_vals, y_vals, timestamps[i]), 6)
        except Exception as e:
            print(f"Lagrange error at index {i}: {e}")
    return result


def newton_fill_missing(timestamps, temperatures):
    temperatures = np.array(temperatures, dtype=np.float64)
    timestamps = np.array(timestamps)
    result = temperatures.copy()
    n = len(temperatures)

    for i in range(n):
        if not np.isnan(temperatures[i]):
            continue

        indices = select_neighbors(i, result, timestamps)
        if indices is None:
            continue

        x_vals = [timestamps[j] for j in indices]
        y_vals = [result[j] for j in indices]

        try:
            coef = newton_divided_differences(x_vals, y_vals)
            result[i] = round(newton_interpolate(x_vals, coef, timestamps[i]), 6)
        except Exception as e:
            print(f"Newton error at index {i}: {e}")
    return result


def process_temperature_data(input_path, output_path, interpolation_method):
    data = pd.read_csv(input_path)
    timestamps = data['Timestamp']
    temperatures = data['Temperature_missing']

    data['Temperature_filled'] = interpolation_method(timestamps, temperatures)
    data['Absolute_Error'] = abs(data['Temperature_filled'] - data['Temperature']).round(6)

    data['Relative_Error (%)'] = data.apply(
        lambda row: round(row['Absolute_Error'] * 100 / row['Temperature'], 6) if row['Temperature'] != 0 else np.nan,
        axis=1
    )

    data.to_csv(output_path, index=False)

process_temperature_data('../data/weather_data.csv', '../data/newton_output.csv', newton_fill_missing)
process_temperature_data('../data/weather_data.csv', '../data/lagrange_output.csv', lagrange_fill_missing)