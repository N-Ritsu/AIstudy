import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- データ生成の設定 ---
N_SAMPLES = 1000  # 総データ点数
N_ANOMALIES = 50  # 異常データ点の数
ANOMALY_MAGNITUDE = 5  # 異常の大きさ

# --- 正常なデータ（サイン波）を生成 ---
time = np.linspace(0, 100, N_SAMPLES)
normal_data = np.sin(time) + np.random.normal(0, 0.2, N_SAMPLES)

# --- 異常データを生成 ---
anomaly_indices = np.random.choice(N_SAMPLES, N_ANOMALIES, replace=False)
anomalous_data = normal_data.copy()
anomalous_data[anomaly_indices] += np.random.normal(0, ANOMALY_MAGNITUDE, N_ANOMALIES)

# --- DataFrameにまとめる ---
df = pd.DataFrame({
    'timestamp': pd.to_datetime(pd.Series(range(N_SAMPLES)), unit='s'),
    'value': anomalous_data,
    'is_anomaly': [1 if i in anomaly_indices else 0 for i in range(N_SAMPLES)]
})

# --- データの保存 ---
df.to_csv('time_series_data.csv', index=False)

print("データセットを作成し、'time_series_data.csv'として保存しました。")
print(f"総データ点数: {N_SAMPLES}")
print(f"異常データ点数: {N_ANOMALIES}")

# --- データの可視化 ---
plt.figure(figsize=(15, 6))
plt.plot(df['timestamp'], df['value'], label='Sensor Value')
anomalies = df[df['is_anomaly'] == 1]
plt.scatter(anomalies['timestamp'], anomalies['value'], color='red', label='Anomaly')
plt.title('Generated Time Series Data with Anomalies')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()