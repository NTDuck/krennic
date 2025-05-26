import pandas as pd
import numpy as np
from datetime import datetime

# Đọc dữ liệu
data = pd.read_csv("real_data.csv")
data["Local Time"] = pd.to_datetime(data["Local Time"])
expected = pd.read_csv("missing.csv")
expected["Local Time"] = pd.to_datetime(expected["Local Time"])

# Hàm spline bậc ba dùng 4 điểm
def cubic_spline_interpolation(x_vals, y_vals, x_target, segment=2):
    x0, x1, x2, x3 = x_vals
    y0, y1, y2, y3 = y_vals

    h1 = (x1 - x0).total_seconds() / 3600
    h2 = (x2 - x1).total_seconds() / 3600
    h3 = (x3 - x2).total_seconds() / 3600

    A = np.array([
        [2*(h1 + h2), h2],
        [h2, 2*(h2 + h3)]
    ])
    b = np.array([
        6 * ((y2 - y1)/h2 - (y1 - y0)/h1),
        6 * ((y3 - y2)/h3 - (y2 - y1)/h2)
    ])
    k2, k3 = np.linalg.solve(A, b)

    if segment == 1:
        h = h1
        a = y0
        c = 0 
        d = (k2 - 2*c) / (6*h)
        b = (y1 - y0)/h - h*(2*c + k2)/6
        dx = (x_target - x0).total_seconds() / 3600

    elif segment == 2:
        h = h2
        a = y1
        c = k2 / 2
        d = (k3 - k2) / (6*h)
        b = (y2 - y1)/h - h*(2*k2 + k3)/6
        dx = (x_target - x1).total_seconds() / 3600

    elif segment == 3:
        h = h3
        a = y2
        c = k3 / 2
        d = -k3 / (6*h) 
        b = (y3 - y2)/h - h*(2*k3 + 0)/6
        dx = (x_target - x2).total_seconds() / 3600

    else:
        raise ValueError("segment must be 1, 2, or 3")

    y_interp = a + b*dx + c*dx**2 + d*dx**3
    return y_interp


errors = []
true = []
cal = []

i = 0
while i < len(data):
    if pd.isna(data.loc[i, "Temperature"]):
        # 2 ô liên tiếp bị thiếu
        if i + 1 < len(data) and pd.isna(data.loc[i + 1, "Temperature"]):
            if i >= 2 and i + 3 < len(data):
                cond = all(pd.notna(data.loc[j, "Temperature"]) for j in [i-2, i-1, i+2, i+3])
                if cond:
                    x_vals1 = pd.to_datetime([
                        data.loc[i-2, "Local Time"],
                        data.loc[i-1, "Local Time"],
                        data.loc[i+2, "Local Time"],
                        data.loc[i+3, "Local Time"]
                    ])
                    y_vals1 = [
                        data.loc[i-2, "Temperature"],
                        data.loc[i-1, "Temperature"],
                        data.loc[i+2, "Temperature"],
                        data.loc[i+3, "Temperature"]
                    ]
                    y1 = cubic_spline_interpolation(x_vals1, y_vals1, data.loc[i, "Local Time"], 2)
                    data.at[i, "Temperature"] = round(y1, 1)

                    y2 = cubic_spline_interpolation(x_vals1, y_vals1, data.loc[i+1, "Local Time"], 2)
                    data.at[i+1, "Temperature"] = round(y2, 1)

                    y_true_1 = expected.loc[expected["Local Time"] == data.loc[i, "Local Time"], "Temperature"].values[0]
                    y_true_2 = expected.loc[expected["Local Time"] == data.loc[i+1, "Local Time"], "Temperature"].values[0]
                    errors.append(abs(y_true_1 - y1))
                    errors.append(abs(y_true_2 - y2))
                    true.append(y_true_1)
                    true.append(y_true_2)
                    cal.append(y1)
                    cal.append(y2)
            i += 2
        else:
            used = False
            if i >= 3 and i + 1 < len(data):
                neighbors = [i-3, i-2, i-1, i+1]
                if all(pd.notna(data.loc[j, "Temperature"]) for j in neighbors):
                    x_vals = pd.to_datetime([data.loc[j, "Local Time"] for j in neighbors])
                    y_vals = [data.loc[j, "Temperature"] for j in neighbors]
                    y = cubic_spline_interpolation(x_vals, y_vals, data.loc[i, "Local Time"], segment=3)
                    used = True

            elif i >= 2 and i + 2 < len(data):
                neighbors = [i-2, i-1, i+1, i+2]
                if all(pd.notna(data.loc[j, "Temperature"]) for j in neighbors):
                    x_vals = pd.to_datetime([data.loc[j, "Local Time"] for j in neighbors])
                    y_vals = [data.loc[j, "Temperature"] for j in neighbors]
                    y = cubic_spline_interpolation(x_vals, y_vals, data.loc[i, "Local Time"], segment=2)
                    used = True

            elif i >= 1 and i + 3 < len(data):  # S3: x[i-1]..x[i+3]
                neighbors = [i-1, i+1, i+2, i+3]
                if all(pd.notna(data.loc[j, "Temperature"]) for j in neighbors):
                    x_vals = pd.to_datetime([data.loc[j, "Local Time"] for j in neighbors])
                    y_vals = [data.loc[j, "Temperature"] for j in neighbors]
                    y = cubic_spline_interpolation(x_vals, y_vals, data.loc[i, "Local Time"], segment=1)
                    used = True

            if used:
                data.at[i, "Temperature"] = round(y, 1)
                y_true = expected.loc[expected["Local Time"] == data.loc[i, "Local Time"], "Temperature"].values[0]
                errors.append(abs(y_true - y))
                true.append(y_true)
                cal.append(y)

            i += 1
    else:
        i += 1


data.to_csv("data_solu.csv", index=False)
print("MAE:", round(np.nanmean(errors), 4))
print("Missing values còn lại:", data["Temperature"].isna().sum())

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(true, cal, color='blue', label='Nội suy vs Thực tế')
plt.plot([min(true), max(true)], [min(true), max(true)], color='red', linestyle='--', label='y = x (hoàn hảo)')

plt.xlabel('Giá trị thực tế (True)')
plt.ylabel('Giá trị nội suy (Interpolated)')
plt.title('Biểu đồ tương quan giữa giá trị thực tế và giá trị nội suy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results.png") 
plt.show()