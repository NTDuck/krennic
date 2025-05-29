import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
data = pd.read_csv("real_data.csv")
data["Local Time"] = pd.to_datetime(data["Local Time"])

# Lọc 3 ngày đầu tiên
data_sorted = data.sort_values("Local Time")  # sắp xếp theo thời gian
first_3_days = data_sorted[data_sorted["Local Time"].dt.date <= data_sorted["Local Time"].dt.date.min() + pd.Timedelta(days=2)]

# Vẽ biểu đồ
plt.figure(figsize=(12, 6))
plt.plot(first_3_days["Local Time"], first_3_days["Temperature"], marker='o', linestyle='-')

# Tùy chỉnh biểu đồ
plt.title("Nhiệt độ theo từng giờ trong 3 ngày đầu")
plt.xlabel("Thời gian")
plt.ylabel("Nhiệt độ (°C)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

plt.show()
